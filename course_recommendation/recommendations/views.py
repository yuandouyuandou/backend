from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import boto3
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

s3_client = boto3.client('s3')
bucket_name = 'cs583-source-data'

class Args:
    maxlen = 50           # Maximum length of user history sequence
    hidden_units = 50     # Dimension of hidden layers
    dropout_rate = 0.2    # Dropout rate
    num_blocks = 2        # Number of Transformer blocks
    num_heads = 5         # Number of heads in multi-head attention
    lr = 0.001            # Learning rate

@csrf_exempt
def recommend_courses(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print(data)
            student_class = data.get("class")  
            student_major = data.get("major")
            student_id = data.get("id")  
            student_interest = data.get("interest", [])  
            courses_taken = data.get("coursesTaken", []) 

            if not isinstance(student_class, int) or not (1 <= student_class <= 5):
                return JsonResponse({"error": "Invalid class value"}, status=400)

            if not isinstance(student_major, int) or not (0 <= student_major < 4):
                return JsonResponse({"error": "Invalid major value"}, status=400)
            elif student_major == 0:
                student_major = 'Applied Mathematics'
            elif student_major == 1:
                student_major = 'Mathematics'
            elif student_major == 2:
                student_major = 'Electrical and Computer Engineering'
            else:
                student_major = 'Computer Science and Engineering'

            if not isinstance(student_interest, list) or len(student_interest) not in [1, 2]:
                return JsonResponse({"error": "Interest must have 1 or 2 items"}, status=400)

            if not isinstance(courses_taken, list):
                return JsonResponse({"error": "coursesTaken should be a list"}, status=400)

            if len(student_interest) == 1:
                student_interest.append('')

            student_data = {
                'id': student_id,
                'history_courses': courses_taken, 
                'interest_1': student_interest[0], 
                'interest_2': student_interest[1], 
                'grade': student_class, 
                'major': student_major
            }

            def download_from_s3(file_key):
                response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
                return response['Body'].read()
            mapping_data = pickle.loads(download_from_s3('mapping_data.pkl'))
            sasrec_weights = download_from_s3('sasrec_weights.weights.h5')


            itemnum = metadata["itemnum"]
            gradenum = metadata["gradenum"]
            vocab_size = metadata["vocab_size"]
            args = metadata["args"]
            course_id_to_idx = metadata["course_id_to_idx"]
            course_data = metadata["course_data"]

            trainer = SASRecTrainer(
                usernum=200, itemnum=itemnum, gradenum=gradenum, vocab_size=vocab_size, args=args, mode="inference"
            )

            # Load pre-trained weights
            trainer.model.load_weights("sasrec_weights")
            recommendations = trainer.recommend(student_data, course_data, num_recommendations=10)

            recommended_course_ids = []
            for student_id, courses in recommendations:
                print(f"Recommendations for student {student_id}:")
                for _, course_id, score in courses:
                    print(f" - Course ID: {course_id}, Score: {score:.4f}")
                    recommended_course_ids.append(course_id)

            return JsonResponse({"recommended_course_ids": recommended_course_ids}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)

    return JsonResponse({"error": "Only POST requests are allowed"}, status=405)

class SASRecTrainer:
    def __init__(self, usernum, itemnum, gradenum, vocab_size, args, mode="inference"):
        """
        Initialize the SASRecTrainer.

        Args:
            usernum: Number of users.
            itemnum: Number of items.
            gradenum: Number of grade levels.
            vocab_size: Vocabulary size for content embedding.
            args: Model hyperparameters.
            mode: 'train' for training mode, 'inference' for inference mode.
        """

        self.model = SASRec(usernum, itemnum, gradenum, vocab_size, args)

        self.args = args
        self.mode = mode

        if mode == "train":
            # Initialize optimizer and loss function in training mode.
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif mode != "inference":
            raise ValueError("Invalid mode. Use 'train' or 'inference'.")

    def train(self, train_inputs_seq, train_inputs_grades, train_inputs_contents, train_pos, train_neg, course_text_indices, max_text_len, epochs, batch_size=64):
        """
        Train the model. Allowed only in 'train' mode.

        Args:
            train_inputs_seq: Sequence of historical courses.
            train_inputs_grades: Grades corresponding to historical courses.
            train_inputs_contents: Content embeddings of historical courses.
            train_pos: Positive samples.
            train_neg: Negative samples.
            course_text_indices: Course content keyword indices.
            max_text_len: Maximum text length for course content.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        if self.mode != "train":
            raise RuntimeError("Training is not allowed in 'inference' mode.")
        
        num_batches = int(len(train_inputs_seq) / batch_size) + 1
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            idx_list = np.arange(len(train_inputs_seq))
            np.random.shuffle(idx_list)
            epoch_loss = 0.0

            for batch_idx in range(num_batches):
                batch_indices = idx_list[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_inputs_seq = train_inputs_seq[batch_indices]
                batch_inputs_grades = train_inputs_grades[batch_indices]
                batch_inputs_contents = train_inputs_contents[batch_indices]
                batch_pos = train_pos[batch_indices]
                batch_neg = train_neg[batch_indices]

                # Prepare positive and negative content indices.
                batch_pos_contents = [
                    course_text_indices.get(cid, [0] * max_text_len) for cid in batch_pos
                ]
                batch_pos_contents = np.array(batch_pos_contents, dtype=np.int32)

                batch_neg_contents = [
                    course_text_indices.get(cid, [0] * max_text_len) for cid in batch_neg
                ]
                batch_neg_contents = np.array(batch_neg_contents, dtype=np.int32)

                # Perform a training step.
                loss = train_step(
                    self.model,
                    batch_inputs_seq,
                    batch_inputs_grades,
                    batch_inputs_contents,
                    batch_pos,
                    batch_pos_contents,
                    batch_neg,
                    batch_neg_contents,
                    self.optimizer,
                    self.loss_fn
                )
                epoch_loss += loss.numpy()

            avg_loss = epoch_loss / num_batches
            print(f"Average loss: {avg_loss:.4f}")

    def recommend(self, student_data, course_data, num_recommendations=10):
        """
        Generate course recommendations.

        Args:
            student_data: Preprocessed student data.
            course_data: Course data for recommendation.
            num_recommendations: Number of recommendations per student.
        """
        return recommend_courses(self.model, student_data, course_data, num_recommendations)