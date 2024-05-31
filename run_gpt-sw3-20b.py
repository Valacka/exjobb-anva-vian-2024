import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import os
from datetime import datetime
import json
import threading


#OUTPUT CONFIG
cpu_mesurments = False   # CPU load measurements
time_mesurments = True   # Processing time measurements
memory_mesurments = False # Memory load measurements

# CPU Information before execution
cpu_usage_data = []
logical_cpus_before = psutil.cpu_count()
physical_cores_before = psutil.cpu_count(logical=False)
cpu_freq_before = psutil.cpu_freq().current

# Initialize Variables
model_name = "AI-Sweden-Models/gpt-sw3-20b-instruct"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
process = psutil.Process(os.getpid())

# Initialize Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()  # Set model to evaluation mode

# Flag to control the monitoring thread
monitoring_active = threading.Event()

def monitor_system_performance():
    process = psutil.Process(os.getpid())  # Get the current process

    while monitoring_active.is_set():
        current_time = time.time()
        cpu_percent = psutil.cpu_percent()
        cpu_times = psutil.cpu_times()
        ctx_switches = process.num_ctx_switches()

        # Record the metrics
        cpu_times_data.append((current_time, cpu_times))
        cpu_usage_data.append((current_time, cpu_percent))
        context_switches_data.append((current_time, ctx_switches))

        time.sleep(0.1)  # Sleep for 100ms


while True:

    json_filename = input("Enter the name of the JSON file to import (or type 'exit' to finish): ")
    if json_filename.lower() == 'exit':
        break

    tasks = []
    try:
        with open(json_filename, 'r') as file:
            tasks = json.load(file)
    except FileNotFoundError:
        print(f"Error: File {json_filename} not found.")
        continue

    iterator = 0
    now = datetime.now()
    print("="*88)
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print("\nTEST ALL GPT-SW3"+" " * 51 , current_time)

    for task in tasks:
        iterator+=1
        # Process the prompt
        prompt = task["prompt"]
        original = task["original"]
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"].to(device)

        # Monitoring before generation
        # Setup and start the monitoring thread
        cpu_times_data = []
        cpu_usage_data = []
        context_switches_data = []
        monitoring_thread = threading.Thread(target=monitor_system_performance)
        monitoring_active.set()  # Signal the thread to start monitoring
        monitoring_thread.start()
        
        memory_usage_before = psutil.virtual_memory().percent
        memory_before = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        start_time = time.time()  # Start timing

        # Generate text
        generated_token_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=task["token_length"],
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=50,
        )[0]

        end_time = time.time()  # Stop timing after generation
        time_taken = end_time - start_time

        # Signal the monitoring thread to stop and wait for it to finish
        monitoring_active.clear()
        monitoring_thread.join()
        # Monitoring after generation
        memory_usage_after = psutil.virtual_memory().percent
        memory_after = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

        # Decode the generated token IDs back to a string, excluding the prompt
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        prompt_length = len(tokenizer.encode(prompt))
        generated_text_without_prompt = tokenizer.decode(generated_token_ids[prompt_length:], skip_special_tokens=True)

        # Print the original and persona-adjusted text and timing
        print("")
        print("-" * 40 +"[TASK "+str(iterator)+"]"+"-" * 40)  # Separator for readability
        print(task["type"])
        print("")
        print("Original Text:", original)
        print("Corrected Text:", generated_text_without_prompt)
        print("")
        
        if(time_mesurments):
            print(f"Time taken: {time_taken} seconds")
            print("")  # Separator for readability
        if(cpu_mesurments):
            print("CPU Times Data:")
            for record in cpu_times_data:
                print(f"Time: {record[0]}, User Time: {record[1].user}, System Time: {record[1].system}")

            print("\nCPU Usage Data:")
            for record in cpu_usage_data:
                print(f"Time: {record[0]}, CPU Usage: {record[1]}%")

            print("\nContext Switches Data:")
            for record in context_switches_data:
                print(f"Time: {record[0]}, Voluntary: {record[1].voluntary}, Involuntary: {record[1].involuntary}")
            print("")  # Separator for readability
        if(memory_mesurments):
            print(f"Total Memory Usage Before: {memory_usage_before}%, After: {memory_usage_after}%")
            print(f"Process Memory Usage Before: {memory_before:.2f} MB, After: {memory_after:.2f} MB")
print("="*88)