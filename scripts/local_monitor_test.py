import psutil, time, random, csv, os

# Create data directory if not exists
os.makedirs("data", exist_ok=True)

# Output CSV file
output_file = "data/local_test_log.csv"

# Open CSV and prepare writer
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "cpu_percent", "mem_percent", "simulated_latency_ms"])

    print("Logging system stats every 0.5s for ~50 seconds...")
    for _ in range(100):  # 100 samples → ~50 seconds
        timestamp = time.time()
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        latency = random.uniform(50, 200)  # fake latency in ms
        writer.writerow([timestamp, cpu, mem, latency])
        time.sleep(0.5)

print(f"Logging complete! Data saved to: {output_file}")
