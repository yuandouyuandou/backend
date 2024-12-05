from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def recommend_courses(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            student_class = data.get("class")  
            student_major = data.get("major")
            student_id = data.get("id")  
            student_interest = data.get("interest", [])  
            courses_taken = data.get("coursesTaken", []) 

            if not isinstance(student_class, int) or not (1 <= student_class <= 5):
                return JsonResponse({"error": "Invalid class value"}, status=400)

            if not isinstance(student_major, int) or not (0 <= student_major < len(MAJOR_CHOICES)):
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

            if not all(interest in dict(INTEREST_CHOICES).keys() for interest in student_interest):
                return JsonResponse({"error": "Invalid interest values"}, status=400)

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

            recommended_course_ids = call_recommendation_model(
                student_class, 
                student_major, 
                student_interest, 
                courses_taken
            )

            return JsonResponse({"recommended_course_ids": recommended_course_ids}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)

    return JsonResponse({"error": "Only POST requests are allowed"}, status=405)