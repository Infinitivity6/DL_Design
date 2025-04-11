# backend/services/task_manager.py

class TaskManager:
    def __init__(self):
        # 用于存储每个任务的状态
        self.task_status = {}

    def add_task(self, task_id, task_info):
        """
        向任务管理器添加一个任务
        """
        if task_id not in self.task_status:
            self.task_status[task_id] = task_info
        else:
            raise ValueError(f"Task ID {task_id} already exists!")

    def update_task(self, task_id, task_info):
        """
        更新已存在任务的状态信息
        """
        if task_id in self.task_status:
            self.task_status[task_id].update(task_info)
        else:
            raise ValueError(f"Task ID {task_id} not found!")

    def get_task(self, task_id):
        """
        获取任务状态信息
        """
        return self.task_status.get(task_id, None)

    def get_all_tasks(self):
        """
        获取所有任务的状态
        """
        return self.task_status
