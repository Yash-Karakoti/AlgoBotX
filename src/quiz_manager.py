# quiz_manager.py
import asyncio
from typing import Dict, Optional
from quiz_data import get_random_questions

class QuizSession:
    def __init__(self, user_id: int, level: str, questions_count: int = 5):
        self.user_id = user_id
        self.level = level
        self.questions = get_random_questions(level, questions_count)
        self.current_question = 0
        self.score = 0
        self.answers = []
        self.start_time = None
        self.is_active = True
        
    def get_current_question(self):
        if self.current_question < len(self.questions):
            return self.questions[self.current_question]
        return None
    
    def answer_question(self, answer: str) -> bool:
        if self.current_question < len(self.questions):
            question = self.questions[self.current_question]
            is_correct = answer.upper() == question["correct"]
            
            self.answers.append({
                "question": question["question"],
                "user_answer": answer.upper(),
                "correct_answer": question["correct"],
                "is_correct": is_correct,
                "explanation": question["explanation"]
            })
            
            if is_correct:
                self.score += 1
            
            self.current_question += 1
            return is_correct
        return False
    
    def is_completed(self) -> bool:
        return self.current_question >= len(self.questions)
    
    def get_results(self):
        total_questions = len(self.questions)
        percentage = (self.score / total_questions) * 100 if total_questions > 0 else 0
        
        return {
            "score": self.score,
            "total": total_questions,
            "percentage": percentage,
            "level": self.level,
            "answers": self.answers
        }

class QuizManager:
    def __init__(self):
        self.active_sessions: Dict[int, QuizSession] = {}
    
    def start_quiz(self, user_id: int, level: str) -> QuizSession:
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
        
        session = QuizSession(user_id, level)
        self.active_sessions[user_id] = session
        return session
    
    def get_session(self, user_id: int) -> Optional[QuizSession]:
        return self.active_sessions.get(user_id)
    
    def end_session(self, user_id: int):
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
    
    def cleanup_inactive_sessions(self):
        # Remove sessions that have been inactive for too long
        inactive_users = []
        for user_id, session in self.active_sessions.items():
            if not session.is_active:
                inactive_users.append(user_id)
        
        for user_id in inactive_users:
            del self.active_sessions[user_id]

# Global quiz manager instance
quiz_manager = QuizManager()