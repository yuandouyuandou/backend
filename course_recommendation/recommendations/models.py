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
