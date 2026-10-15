import unittest
from unittest.mock import patch, mock_open
import json
import os
from Jarwis import (
    detect_intent,
    remember_fact,
    recall_fact,
    project_status,
    add_task,
    complete_task,
    load_memory,
    save_memory,
    load_tasks,
    save_tasks
)

class TestAgFunctions(unittest.TestCase):
    def setUp(self):
        # Mock memory file operations
        self.memory_patcher = patch('builtins.open', mock_open())
        self.memory_patcher.start()
        
        # Mock tasks file operations
        self.tasks_patcher = patch('Jarwis.open', mock_open())
        self.tasks_patcher.start()
        
        # Sample memory structure
        self.sample_memory = {
            "general": {"me": "developer"},
            "projects": {},
            "vehicles": {},
            "characters": {},
            "notes": {},
            "tasks": {}
        }
        
        # Sample tasks structure
        self.sample_tasks = {
            "active": ["task1", "task2"],
            "completed": ["task3"]
        }

    def tearDown(self):
        self.memory_patcher.stop()
        self.tasks_patcher.stop()

    def test_detect_intent(self):
        self.assertEqual(detect_intent("hello"), "greeting")
        self.assertEqual(detect_intent("project status"), "project_status")
        self.assertEqual(detect_intent("what is gravity"), "recall")
        self.assertEqual(detect_intent("add task write tests"), "add_task")
        self.assertEqual(detect_intent("random input"), "unknown")

    def test_remember_fact(self):
        memory = self.sample_memory.copy()
        result = remember_fact("remember I am a developer", memory)
        self.assertEqual(result, "Jarwis: Memory stored in general. You are a developer.")
        self.assertEqual(memory["general"]["me"], "a developer")
        
        result = remember_fact("remember color is blue", memory)
        self.assertEqual(result, "Jarwis: Memory stored in general. Color is blue.")
        self.assertEqual(memory["general"]["color"], "blue")

    def test_recall_fact(self):
        memory = self.sample_memory.copy()
        self.assertEqual(recall_fact("who am I", memory), "Jarwis: You are developer. Stored in general.")
        self.assertEqual(recall_fact("what is me", memory), "Jarwis: You are developer. Stored in general.")
        self.assertEqual(recall_fact("what is color", memory), "Jarwis: I do not have memory of 'color'. Yet.")

    @patch('Jarwis.load_project_context')
    def test_project_status(self, mock_load_project):
        mock_load_project.return_value = {
            "full_name": "Test Project",
            "project_name": "TP",
            "current_version": "1.0",
            "current_phase": "Testing",
            "current_objective": "Write tests",
            "completed_milestones": ["Design", "Implementation"],
            "next_milestone": "Deployment",
            "development_rule": "Test everything"
        }
        
        expected = (
            "Project Status\n\n"
            "Project: Test Project\n"
            "Short Name: TP\n"
            "Version: 1.0\n"
            "Phase: Testing\n"
            "Objective: Write tests\n\n"
            "Completed Milestones: 2\n"
            "Next Milestone: Deployment\n\n"
            "Rule: Test everything"
        )
        
        self.assertEqual(project_status(), expected)

    def test_add_task(self):
        tasks = self.sample_tasks.copy()
        result = add_task("add task write unit tests", tasks)
        self.assertEqual(result, "Jarwis: Task added. Active tasks: 3.")
        self.assertIn("write unit tests", tasks["active"])
        
        result = add_task("remind me to refactor code", tasks)
        self.assertEqual(result, "Jarwis: Task added. Active tasks: 4.")
        self.assertIn("refactor code", tasks["active"])

    def test_complete_task(self):
        tasks = self.sample_tasks.copy()
        result = complete_task("complete task 1", tasks)
        self.assertEqual(result, "Jarwis: Task completed: task1. Progress detected. Rare, but welcome.")
        self.assertNotIn("task1", tasks["active"])
        self.assertIn("task1", tasks["completed"])
        
        result = complete_task("mark task number 1 done", tasks)
        self.assertEqual(result, "Jarwis: Task completed: task2. Progress detected. Rare, but welcome.")
        self.assertEqual(len(tasks["active"]), 0)

if __name__ == '__main__':
    unittest.main()
