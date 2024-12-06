from django.db import models
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Student(models.Model):
    GRADE_CHOICES = [(i, f'Grade {i}') for i in range(1, 6)]
    MAJOR_CHOICES = [('amath', 'Applied Mathematics'), ('math', 'Mathematics'), ('ece', 'Electrical and Computer Engineering'), ('cse', 'Computer Science and Engineering')]
    INTEREST_CHOICES = [
    ('computation', 'Computation & Scientific Computing'),
    ('differential_eq', 'Differential Equations & Dynamical Systems'),
    ('optimization', 'Optimization'),
    ('algebra', 'Algebra & Abstract Algebra'),
    ('probability', 'Probability & Stochastic Processes'),
    ('topology', 'Topology & Geometry'),
    ('analysis', 'Analysis & Real Analysis'),
    ('complex_analysis', 'Complex Analysis & Complex Variables'),
    ('math_modeling', 'Mathematical Modeling'),
    ('math_biology', 'Mathematical Biology'),
    ('numerical_analysis', 'Numerical Analysis'),
    ('combinatorics', 'Combinatorics & Discrete Mathematics'),
    ('computer_math', 'Computer-Aided Mathematics'),
    ('discrete_math', 'Discrete Mathematics & Graph Theory'),
    ('math_education', 'Mathematics Education & Career Development'),
    ('hardware_architecture', 'Computer Hardware & Architecture'),
    ('signal_processing', 'Signal Processing & Communication'),
    ('control_systems', 'Control Systems & Optimization'),
    ('embedded_systems', 'Embedded & Real-Time Systems'),
    ('bioengineering', 'Bioengineering & Neural Engineering'),
    ('quantum_computing', 'Quantum Computing & Information'),
    ('computer_vision', 'Computer Vision & Image Processing'),
    ('ai_ml', 'Artificial Intelligence & Machine Learning'),
    ('data_science', 'Data Science & Data Engineering'),
    ('hci_ux', 'Human-Computer Interaction & UI/UX'),
    ('networks_security', 'Computer Networks & Security'),
    ('software_dev', 'Software Development & Engineering'),
    ('database_management', 'Database & Data Management'),
    ('robotics', 'Robotics & Automation'),
    ('power_systems', 'Power Systems & Energy Engineering')
    ]

    grade = models.IntegerField(choices=GRADE_CHOICES, null=True, blank=True)
    major = models.CharField(max_length=20, choices=MAJOR_CHOICES, null=True, blank=True)
    interest = models.CharField(max_length=80, choices=INTEREST_CHOICES, null=True, blank=True)

    def __str__(self):
        return f"Student ({self.grade}, {self.major}, {self.interest})"

class Course(models.Model):
    course_id = models.IntegerField(unique=True)
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Args:
    maxlen = 50           # Maximum length of user history sequence
    hidden_units = 50     # Dimension of hidden layers
    dropout_rate = 0.2    # Dropout rate
    num_blocks = 2        # Number of Transformer blocks
    num_heads = 5         # Number of heads in multi-head attention
    lr = 0.001            # Learning rate

args = Args()

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
    
class SASRec(tf.keras.Model):
    def __init__(self, usernum, itemnum, gradenum, vocab_size, args):
        super(SASRec, self).__init__()
        self.args = args
        self.item_emb = tf.keras.layers.Embedding(input_dim=itemnum + 1, output_dim=args.hidden_units, mask_zero=True)
        self.grade_emb = tf.keras.layers.Embedding(input_dim=gradenum, output_dim=args.hidden_units, mask_zero=True)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=args.maxlen, output_dim=args.hidden_units * 3)
        self.dropout = tf.keras.layers.Dropout(args.dropout_rate)
        self.attention_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=args.num_heads, key_dim=args.hidden_units * 3) for _ in range(args.num_blocks)
        ]
        self.ffn_layers = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(args.hidden_units * 3, activation='relu'),
                tf.keras.layers.Dense(args.hidden_units * 3)
            ]) for _ in range(args.num_blocks)
        ]
        self.layernorms = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(2 * args.num_blocks)]
        self.final_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Add course content embedding layer
        self.content_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=args.hidden_units, mask_zero=True)
    
    def call(self, inputs, training):
        seq = inputs['seq']             # (batch_size, seq_len)
        grades = inputs['grades']       # (batch_size, seq_len)
        contents = inputs['contents']   # (batch_size, seq_len, max_text_len)

        # Build mask for the sequence
        mask = tf.cast(tf.not_equal(seq, 0), tf.float32)[:, :, tf.newaxis]

        # Course ID embeddings and grade embeddings
        seq_emb = self.item_emb(seq)      # (batch_size, seq_len, hidden_units)
        grade_emb = self.grade_emb(grades)  # (batch_size, seq_len, hidden_units)

        # Course content embeddings
        batch_size, seq_len, text_len = contents.shape
        contents_flat = tf.reshape(contents, [batch_size * seq_len, text_len])  # (batch_size * seq_len, max_text_len)
        content_emb_flat = self.content_emb(contents_flat)  # (batch_size * seq_len, max_text_len, hidden_units)
        # Average the text embeddings for each course
        content_emb_flat = tf.reduce_mean(content_emb_flat, axis=1)  # (batch_size * seq_len, hidden_units)
        content_emb = tf.reshape(content_emb_flat, [batch_size, seq_len, self.args.hidden_units])  # (batch_size, seq_len, hidden_units)

        # Concatenate embedding vectors
        seq_emb = tf.concat([seq_emb, grade_emb, content_emb], axis=-1)  # (batch_size, seq_len, hidden_units * 3)

        # Positional encoding
        pos_indices = tf.range(tf.shape(seq)[1])  # (seq_len,)
        pos_emb = self.pos_emb(pos_indices)       # (seq_len, hidden_units * 3)
        seq_emb += pos_emb  # (batch_size, seq_len, hidden_units * 3)

        # Dropout
        seq_emb = self.dropout(seq_emb, training=training)
        seq_emb *= mask  # Apply mask

        # Multi-layer self-attention and feed-forward network
        for i in range(self.args.num_blocks):
            # Multi-head self-attention
            seq_emb_norm = self.layernorms[2 * i](seq_emb)
            attn_output = self.attention_layers[i](seq_emb_norm, seq_emb_norm, attention_mask=None)
            attn_output = self.dropout(attn_output, training=training)
            seq_emb += attn_output
            seq_emb *= mask  # Apply mask

            # Feed-forward network
            seq_emb_norm = self.layernorms[2 * i + 1](seq_emb)
            ffn_output = self.ffn_layers[i](seq_emb_norm)
            ffn_output = self.dropout(ffn_output, training=training)
            seq_emb += ffn_output
            seq_emb *= mask  # Apply mask

        # Final layer normalization
        seq_emb = self.final_layernorm(seq_emb)  # (batch_size, seq_len, hidden_units * 3)

        return seq_emb  # Return sequence representation for prediction

# 5. Build training data and testing data (including course grade information and content embeddings)

def generate_data(student_data, course_grades, course_text_indices, args, max_text_len, is_train=True):
    """
    Generate training or testing datasets.
    
    Args:
        student_data: List of student information.
        course_grades: Mapping of course grades.
        course_text_indices: Mapping of course content embeddings (keyword indices).
        args: Model hyperparameters.
        max_text_len: Maximum length for course content keywords.
        is_train: Boolean, True for generating training data, False for testing data.

    Returns:
        inputs_seq: Historical course sequences.
        inputs_grades: Corresponding course grade sequences.
        inputs_contents: Corresponding course content embeddings.
        pos: Positive samples.
        neg: Negative samples (only for training data).
    """
    inputs_seq = []
    inputs_grades = []
    inputs_contents = []
    pos = []
    neg = []

    for student in student_data:
        history = student['history_courses']
        if len(history) < 2:  # At least two courses are required (history + target)
            continue

        if is_train:
            # Training data: Use all courses except the last one to generate samples
            for i in range(1, len(history) - 1):
                seq = history[:i]
                pos_item = history[i]
                neg_item = np.random.choice([cid for cid in course_data['course_ids'] if cid not in history])  # Negative sample

                # Build grade and content embeddings
                seq_grades = []
                seq_contents = []
                for cid in seq:
                    grade = course_grades.get(idx_to_course_id[cid], 0)
                    grade_idx = grade_to_idx.get(grade, 0)
                    seq_grades.append(grade_idx)

                    content_indices = course_text_indices.get(cid, [0] * max_text_len)
                    seq_contents.append(content_indices)

                # Pad sequences to fixed length
                seq_padded = [0] * (args.maxlen - len(seq)) + seq[-args.maxlen:]
                seq_grades_padded = [0] * (args.maxlen - len(seq_grades)) + seq_grades[-args.maxlen:]
                seq_contents_padded = [[0] * max_text_len] * (args.maxlen - len(seq_contents)) + seq_contents[-args.maxlen:]

                inputs_seq.append(seq_padded)
                inputs_grades.append(seq_grades_padded)
                inputs_contents.append(seq_contents_padded)
                pos.append(pos_item)
                neg.append(neg_item)

        else:
            # Testing data: Use the first (len(history) - 1) courses to predict the last one
            seq = history[:-1]  # First n-1 courses as input
            pos_item = history[-1]  # Last course as the target

            # Build grade and content embeddings
            seq_grades = []
            seq_contents = []
            for cid in seq:
                grade = course_grades.get(idx_to_course_id[cid], 0)
                grade_idx = grade_to_idx.get(grade, 0)
                seq_grades.append(grade_idx)

                content_indices = course_text_indices.get(cid, [0] * max_text_len)
                seq_contents.append(content_indices)

            # Pad sequences to fixed length
            seq_padded = [0] * (args.maxlen - len(seq)) + seq[-args.maxlen:]
            seq_grades_padded = [0] * (args.maxlen - len(seq_grades)) + seq_grades[-args.maxlen:]
            seq_contents_padded = [[0] * max_text_len] * (args.maxlen - len(seq_contents)) + seq_contents[-args.maxlen:]

            inputs_seq.append(seq_padded)
            inputs_grades.append(seq_grades_padded)
            inputs_contents.append(seq_contents_padded)
            pos.append(pos_item)

    if is_train:
        return (
            np.array(inputs_seq, dtype=np.int32),
            np.array(inputs_grades, dtype=np.int32),
            np.array(inputs_contents, dtype=np.int32),
            np.array(pos, dtype=np.int32),
            np.array(neg, dtype=np.int32)
        )
    else:
        return (
            np.array(inputs_seq, dtype=np.int32),
            np.array(inputs_grades, dtype=np.int32),
            np.array(inputs_contents, dtype=np.int32),
            np.array(pos, dtype=np.int32)
        )

# 6. Define training and recommendation functions
# Helper function: calculate keyword match score
def calculate_keyword_match_score(course_keywords, student_interest):
    course_keywords = str(course_keywords).strip()
    student_interest = str(student_interest).strip()
    if not course_keywords or not student_interest:
        return 0.0  # Return 0 if keywords or interests are empty
    vectorizer = CountVectorizer().fit_transform([course_keywords, student_interest])
    vectors = vectorizer.toarray()
    if vectors.shape[1] == 0:
        return 0.0  # Return 0 if no common keywords
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]

# Training step function (modified to include course content embeddings)
@tf.function
def train_step(model, inputs_seq, inputs_grades, inputs_contents, pos, pos_contents, neg, neg_contents, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        inputs = {'seq': inputs_seq, 'grades': inputs_grades, 'contents': inputs_contents}
        seq_emb = model(inputs, training=True)
        seq_emb_last = seq_emb[:, -1, :]  # (batch_size, hidden_units * 3)

        # Positive sample embeddings
        pos_grade = tf.gather(course_idx_to_grade_idx_tf, pos)
        pos_emb = model.item_emb(pos)
        pos_grade_emb = model.grade_emb(pos_grade)
        pos_content_emb = model.content_emb(pos_contents)
        pos_content_emb = tf.reduce_mean(pos_content_emb, axis=1)
        pos_emb = tf.concat([pos_emb, pos_grade_emb, pos_content_emb], axis=-1)

        # Negative sample embeddings
        neg_grade = tf.gather(course_idx_to_grade_idx_tf, neg)
        neg_emb = model.item_emb(neg)
        neg_grade_emb = model.grade_emb(neg_grade)
        neg_content_emb = model.content_emb(neg_contents)
        neg_content_emb = tf.reduce_mean(neg_content_emb, axis=1)
        neg_emb = tf.concat([neg_emb, neg_grade_emb, neg_content_emb], axis=-1)

        # Compute logits
        pos_logits = tf.reduce_sum(seq_emb_last * pos_emb, axis=-1)
        neg_logits = tf.reduce_sum(seq_emb_last * neg_emb, axis=-1)

        # Compute loss
        istarget = tf.cast(tf.not_equal(pos, 0), tf.float32)
        loss = tf.reduce_sum(
            loss_fn(tf.ones_like(pos_logits), pos_logits) * istarget +
            loss_fn(tf.zeros_like(neg_logits), neg_logits) * istarget
        ) / tf.reduce_sum(istarget)

    # Apply gradients
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Recommendation generation function (modified to include course content embeddings)
def recommend_courses(model, student_data, course_data, num_recommendations=10):
    recommendations = []
    for student in student_data:
        history_courses = student['history_courses']  # Already a list of integer indices
        if not history_courses:
            print(f"Student {student['id']} has no valid history courses.")
            continue  # Skip if no history courses
        interest_keywords = " ".join([str(student['interest_1']), str(student['interest_2'])])

        # Convert history sequence to fixed length (padding or truncating)
        maxlen = model.args.maxlen
        if len(history_courses) >= maxlen:
            history_seq = history_courses[-maxlen:]
        else:
            history_seq = [0] * (maxlen - len(history_courses)) + history_courses

        # Get corresponding grades and content indices
        history_grades = []
        history_contents = []
        for cid in history_seq:
            if cid == 0:
                history_grades.append(0)
                history_contents.append([0] * max_text_len)
            else:
                course_id = idx_to_course_id[cid]
                grade = course_grades.get(course_id, 0)
                grade_idx = grade_to_idx.get(grade, 0)
                history_grades.append(grade_idx)

                content_indices = course_text_indices.get(cid, [0] * max_text_len)
                history_contents.append(content_indices)

        # Convert to NumPy arrays with integer data type
        history_seq = np.array(history_seq, dtype=np.int32).reshape(1, -1)
        history_grades = np.array(history_grades, dtype=np.int32).reshape(1, -1)
        history_contents = np.array(history_contents, dtype=np.int32).reshape(1, maxlen, max_text_len)

        inputs = {'seq': history_seq, 'grades': history_grades, 'contents': history_contents}

        # Use SASRec model to get sequence representation
        seq_emb = model(inputs, training=False)  # (1, seq_len, hidden_units * 3)
        seq_emb = seq_emb[:, -1, :]  # Take output from the last time step (1, hidden_units * 3)

        # Get embeddings for candidate courses
        candidate_course_indices = np.array(course_data['course_ids'], dtype=np.int32)
        candidate_grades = course_idx_to_grade_idx[candidate_course_indices]
        candidate_emb = model.item_emb(candidate_course_indices)  # (num_items, hidden_units)
        candidate_grade_emb = model.grade_emb(candidate_grades)   # (num_items, hidden_units)

        # Get content embeddings for candidate courses
        candidate_contents = []
        for cid in candidate_course_indices:
            content_indices = course_text_indices.get(cid, [0] * max_text_len)
            candidate_contents.append(content_indices)
        candidate_contents = np.array(candidate_contents, dtype=np.int32)  # (num_items, max_text_len)
        candidate_content_emb = model.content_emb(candidate_contents)  # (num_items, max_text_len, hidden_units)
        candidate_content_emb = tf.reduce_mean(candidate_content_emb, axis=1)  # (num_items, hidden_units)

        # Concatenate embedding vectors
        candidate_emb = tf.concat([candidate_emb, candidate_grade_emb, candidate_content_emb], axis=-1)  # (num_items, hidden_units * 3)

        # Compute scores
        scores = tf.matmul(seq_emb, candidate_emb, transpose_b=True).numpy().flatten()  # (num_items,)

        # Compute keyword match score and combine
        final_scores = []
        for i, course_idx in enumerate(candidate_course_indices):
            course_keywords = course_data['keywords'].get(course_idx, '')
            match_score = calculate_keyword_match_score(course_keywords, interest_keywords)
            final_score = 0.7 * scores[i] + 0.3 * match_score  # Weighted sum
            final_scores.append((student['id'], idx_to_course_id[course_idx], final_score))  # Map back to course ID

        # Add to recommendation list
        final_scores.sort(key=lambda x: x[2], reverse=True)
        recommendations.append((student['id'], final_scores[:num_recommendations]))
    
    return recommendations

# 7. Train the model
# Define optimizer and loss function

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

def process_student_data(merged_data, course_id_to_idx):

    student_data = []

    for _, row in merged_data.iterrows():
        history_course_ids = [cid.strip().upper(cid) for cid in str(row['Courses']).split(', ')]

        history_courses = []
        for cid in history_course_ids:
            if cid in course_id_to_idx:
                history_courses.append(course_id_to_idx[cid])
            else:
                print(f"Unmapped course ID {cid} for student {row['StudentID']}")

        student_entry = {
            'id': row['StudentID'],
            'history_courses': history_courses,
            'interest_1': row.get('Interest_1', ''),
            'interest_2': row.get('Interest_2', ''),
            'grade': row.get('Grade', ''),
            'major': row.get('Major', '')
        }
        student_data.append(student_entry)

    return student_data