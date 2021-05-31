"""
Simple wrapper to compute a parrallelize a function.

given a function f that takes a single argument, `arg`, the batch_job decorator
turns f into a function that takes a list of args and computes each result in
a seperate process. It then returns the results as a list.

___

use:
    ```
    batch_job = BatchJob(num_processes=3)

    @batch_job
    def job(arg):
        # do something...

    job([args_1, args_2, ....])
    ```

The above runs: job(args_1), job(args_2), ... in parrallel and returns
[job(args_1), job(args_2), ...]
"""


from multiprocessing import Process, Queue, JoinableQueue, cpu_count

PROCESSES = cpu_count() - 1


class BatchJob:
    def __init__(self, num_processes=PROCESSES):
        self.in_queue = JoinableQueue()
        self.out_queue = Queue()
        self.num_processes = num_processes

    def __call__(self, job):
        def batched_job(arg_batch):
            def process_task(task_queue, return_queue):
                while True:
                    next_task = task_queue.get()
                    if next_task is None:
                        task_queue.task_done()
                        break
                    args_ind, args = next_task
                    result = job(args)
                    task_queue.task_done()
                    return_queue.put((args_ind, result))
                return True

            for i in range(self.num_processes):
                p = Process(target=process_task,
                            args=(self.in_queue, self.out_queue))
                p.start()

            for ind, arg in enumerate(arg_batch):
                self.in_queue.put((ind, arg))

            for _ in range(self.num_processes):
                self.in_queue.put(None)

            self.in_queue.join()

            results = []
            while not self.out_queue.empty():
                results.append(self.out_queue.get())
            results.sort(key=lambda item: item[0])
            return [r for i, r in results]
        return batched_job
