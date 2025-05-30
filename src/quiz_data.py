# quiz_data.py
import random

QUIZ_QUESTIONS = {
    "beginner": [
        {
            "question": "What is Algorand's consensus mechanism called?",
            "options": ["A) Proof of Work", "B) Pure Proof of Stake", "C) Delegated Proof of Stake", "D) Proof of Authority"],
            "correct": "B",
            "explanation": "Algorand uses Pure Proof of Stake (PPoS) consensus mechanism."
        },
        {
            "question": "What is the native cryptocurrency of Algorand?",
            "options": ["A) ALGO", "B) ADA", "C) ETH", "D) BTC"],
            "correct": "A",
            "explanation": "ALGO is the native cryptocurrency of the Algorand blockchain."
        },
        {
            "question": "How long does it take for a transaction to be finalized on Algorand?",
            "options": ["A) 10 minutes", "B) 1 minute", "C) 4.5 seconds", "D) 30 seconds"],
            "correct": "C",
            "explanation": "Algorand transactions are finalized in approximately 4.5 seconds."
        },
        {
            "question": "What is the minimum ALGO balance required to keep an account active?",
            "options": ["A) 0.1 ALGO", "B) 1 ALGO", "C) 0.001 ALGO", "D) 10 ALGO"],
            "correct": "A",
            "explanation": "The minimum balance requirement is 0.1 ALGO to keep an account active."
        },
        {
            "question": "What programming language is primarily used for Algorand smart contracts?",
            "options": ["A) Solidity", "B) Python", "C) TEAL", "D) JavaScript"],
            "correct": "C",
            "explanation": "TEAL (Transaction Execution Approval Language) is used for Algorand smart contracts."
        }
    ],
    "intermediate": [
        {
            "question": "What is the maximum transaction throughput of Algorand?",
            "options": ["A) 1,000 TPS", "B) 10,000 TPS", "C) 46,000 TPS", "D) 100,000 TPS"],
            "correct": "C",
            "explanation": "Algorand can handle up to 46,000 transactions per second."
        },
        {
            "question": "What is an ASA in Algorand?",
            "options": ["A) Algorand Smart Account", "B) Algorand Standard Asset", "C) Algorand Secure Application", "D) Algorand State Array"],
            "correct": "B",
            "explanation": "ASA stands for Algorand Standard Asset, which represents tokens on Algorand."
        },
        {
            "question": "What is the role of participation keys in Algorand?",
            "options": ["A) Transaction signing", "B) Consensus participation", "C) Account creation", "D) Asset transfer"],
            "correct": "B",
            "explanation": "Participation keys are used to participate in the Algorand consensus protocol."
        },
        {
            "question": "What is the maximum size of a single transaction group in Algorand?",
            "options": ["A) 8 transactions", "B) 16 transactions", "C) 32 transactions", "D) 64 transactions"],
            "correct": "B",
            "explanation": "A transaction group can contain up to 16 atomic transactions."
        },
        {
            "question": "What is the purpose of the StateProof in Algorand?",
            "options": ["A) Transaction validation", "B) Cross-chain interoperability", "C) Smart contract execution", "D) Account verification"],
            "correct": "B",
            "explanation": "StateProofs enable secure cross-chain communication and interoperability."
        }
    ],
    "advanced": [
        {
            "question": "What is the VRF (Verifiable Random Function) used for in Algorand?",
            "options": ["A) Transaction ordering", "B) Leader selection", "C) Fee calculation", "D) Block validation"],
            "correct": "B",
            "explanation": "VRF is used for cryptographically secure and verifiable leader selection in consensus."
        },
        {
            "question": "What is the maximum number of inner transactions in a single application call?",
            "options": ["A) 16", "B) 32", "C) 256", "D) 1024"],
            "correct": "C",
            "explanation": "A single application call can create up to 256 inner transactions."
        },
        {
            "question": "What is the box storage feature in Algorand used for?",
            "options": ["A) Large data storage in smart contracts", "B) Transaction batching", "C) Key management", "D) Asset creation"],
            "correct": "A",
            "explanation": "Box storage allows smart contracts to store large amounts of data efficiently."
        },
        {
            "question": "What is the difference between LocalState and GlobalState in Algorand smart contracts?",
            "options": ["A) LocalState is per-user, GlobalState is per-application", "B) LocalState is temporary, GlobalState is permanent", "C) LocalState is encrypted, GlobalState is public", "D) No difference"],
            "correct": "A",
            "explanation": "LocalState stores data per user account, while GlobalState stores data per application."
        },
        {
            "question": "What is the AVM (Algorand Virtual Machine) responsible for?",
            "options": ["A) Transaction processing", "B) Smart contract execution", "C) Consensus algorithm", "D) Network communication"],
            "correct": "B",
            "explanation": "AVM executes smart contracts written in TEAL on the Algorand blockchain."
        }
    ]
}

def get_random_questions(level: str, count: int = 5):
    """Get random questions for a specific level"""
    questions = QUIZ_QUESTIONS.get(level, [])
    return random.sample(questions, min(count, len(questions)))