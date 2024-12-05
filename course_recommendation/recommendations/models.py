from django.db import models

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