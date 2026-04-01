from pydantic import BaseModel

class TaskInfo(BaseModel):
    task_id: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

TaskInfo(task_id=1)
