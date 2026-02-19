"""
上传数据 + 启动 OpenAI fine-tuning job。
"""
import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_FILE = "./training_data/openai-finetune.jsonl"
MODEL = "gpt-4o-mini-2024-07-18"
SUFFIX = "joker-bot"


def main():
    # 1. 上传文件
    print("上传训练数据...")
    with open(DATA_FILE, "rb") as f:
        file_obj = client.files.create(file=f, purpose="fine-tune")
    print(f"  文件 ID: {file_obj.id}")
    print(f"  文件名: {file_obj.filename}")
    print(f"  大小: {file_obj.bytes:,} bytes")

    # 等文件处理完
    print("等待文件处理...")
    while True:
        file_status = client.files.retrieve(file_obj.id)
        if file_status.status == "processed":
            print("  文件处理完成")
            break
        elif file_status.status == "error":
            print(f"  文件处理失败: {file_status.status_details}")
            sys.exit(1)
        time.sleep(2)

    # 2. 创建 fine-tuning job
    print(f"\n启动 fine-tuning (模型: {MODEL})...")
    job = client.fine_tuning.jobs.create(
        training_file=file_obj.id,
        model=MODEL,
        suffix=SUFFIX,
        hyperparameters={
            "n_epochs": 3,
        },
    )
    print(f"  Job ID: {job.id}")
    print(f"  状态: {job.status}")
    print(f"\n训练已提交！通常需要 15-60 分钟。")
    print(f"查看进度: https://platform.openai.com/finetune/{job.id}")

    # 3. 轮询状态
    print("\n等待训练完成...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        status = job.status

        if status == "succeeded":
            print(f"\n训练完成！")
            print(f"  Fine-tuned 模型: {job.fine_tuned_model}")
            print(f"  训练 token: {job.trained_tokens:,}")

            # 保存模型名到文件
            with open("./fine_tuned_model.txt", "w") as f:
                f.write(job.fine_tuned_model)
            print(f"  模型名已保存到 fine_tuned_model.txt")
            break

        elif status in ("failed", "cancelled"):
            print(f"\n训练失败: {status}")
            if job.error:
                print(f"  错误: {job.error}")
            sys.exit(1)

        else:
            # 获取最新事件
            events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=1)
            latest = events.data[0].message if events.data else "..."
            print(f"  [{status}] {latest}")
            time.sleep(30)


if __name__ == "__main__":
    main()
