import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import os
from datetime import datetime

#OUTPUT CONFIG
cpu_mesurments = True   # CPU load mesurments
time_mesurments = True   # Processing time mesurments
memory_mesurments = True # Memory load mesurments

# CPU Information before execution
logical_cpus_before = psutil.cpu_count()
physical_cores_before = psutil.cpu_count(logical=False)
cpu_freq_before = psutil.cpu_freq().current

# Initialize Variables
model_name = "AI-Sweden-Models/gpt-sw3-20b-instruct"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
process = psutil.Process(os.getpid())

# Define multiple prompts with intentional spelling errors
tasks = [
    {
        "prompt": "Correct the spelling errors in the following Swedish sentence, Example: 'Denna twxt är felstavad.' should be corrected to 'Denna text är felstavat.' Now correct this in Swedish: 'På en solig dag bestämde sig Anna och hennes vän Karin för att ta en piknik i den närligande parken. De packade en korg med mackor, fruckt, saft och kakor. Väl framme bredde de ut en filt på marken och njöt av den vackra utsickten medan de åt.' to: '",
        "original": "På en solig dag bestämde sig Anna och hennes vän Karin för att ta en piknik i den närligande parken. De packade en korg med mackor, fruckt, saft och kakor. Väl framme bredde de ut en filt på marken och njöt av den vackra utsickten medan de åt.",
        "token_length": 58
    },
    {
        "prompt": "Correct the spelling errors in the following Swedish sentence, Example: 'Vi har en viktg möte imorgon.' should be corrected to 'Vi har ett viktigt möte imorgon.' Now correct this in Swedish: 'Det är viktgt att alla dokument är ordntligt granskade innan inlämning.' to: '",
        "original": "Det är viktgt att alla dokument är ordntligt granskade innan inlämning.",
        "token_length": 13
    },
    {
        "prompt": "Correct the spelling errors in the following Swedish sentence, Example: 'Kan du hjäla mig med detta?' should be corrected to 'Kan du hjälpa mig med detta?' Now correct this in Swedish: 'Jag behöver din hjälp för att förberdea presentationen. Dom vil att den ska vara ferdig imorgon' to: '",
        "original": "Jag behöver din hjälp för att förberdea presentationen.",
        "token_length": 12
    },
    {
        "prompt": "Correct the spelling errors in the following Swedish sentence, Example: 'Han är en duktg programmerare.' should be corrected to 'Han är en duktig programmerare.' Now correct this in Swedish: 'Vi måste förbätra vår kodkvalitet och testnig processer.' to: '",
        "original": "Vi måste förbätra vår kodkvalitet och testnig processer.",
        "token_length": 12
    }
]

# Initialize Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()  # Set model to evaluation mode

iterator = 0
now = datetime.now()
print("="*88)
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
print("\nSPELLING TEST GPT-SW3"+" " * 47 , current_time)

for task in tasks:
    iterator+=1
    # Process the prompt
    prompt = task["prompt"]
    original = task["original"]
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"].to(device)

    # Monitoring before generation
    cpu_usage_before = psutil.cpu_percent(interval=1)
    percpu_usage_before = psutil.cpu_percent(interval=1, percpu=True)
    memory_usage_before = psutil.virtual_memory().percent
    memory_before = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    start_time = time.time()  # Start timing

    # Generate corrected text
    generated_token_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=task["token_length"],  # Adjusted to a reasonable length for the corrected text
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
    )[0]  # Take the first (and only) generated sequence

    # Monitoring after generation
    cpu_usage_after = psutil.cpu_percent(interval=1)
    percpu_usage_after = psutil.cpu_percent(interval=1, percpu=True)
    memory_usage_after = psutil.virtual_memory().percent
    memory_after = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    end_time = time.time()  # Stop timing after generation
    time_taken = end_time - start_time

    # Decode the generated token IDs back to a string
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    prompt_length = len(tokenizer.encode(prompt))
    generated_text_without_prompt = tokenizer.decode(generated_token_ids[prompt_length:], skip_special_tokens=True)

    # Print the corrected text and timing
    print("")
    print("-" * 40 +"[TASK "+str(iterator)+"]"+"-" * 40)  # Separator for readability
    print("")
    print("Original Text with Errors:", original)
    print("")
    print("Corrected Text:", generated_text_without_prompt)
    print("")  # Separator for readability

    if(time_mesurments):
        print(f"Time taken: {time_taken} seconds")
        print("")  # Separator for readability
    if(cpu_mesurments):
        print(f"CPU Usage Before: {cpu_usage_before}%, After: {cpu_usage_after}%")
        print(f"Per-CPU Usage Before: {percpu_usage_before}, After: {percpu_usage_after}")
        print(f"Logical CPUs: {logical_cpus_before}, Physical Cores: {physical_cores_before}")
        print(f"CPU Frequency: {cpu_freq_before}MHz")
        print("")  # Separator for readability
    if(memory_mesurments):
        print(f"Total Memory Usage Before: {memory_usage_before}%, After: {memory_usage_after}%")
        print(f"Process Memory Usage Before: {memory_before:.2f} MB, After: {memory_after:.2f} MB")
print("="*88)