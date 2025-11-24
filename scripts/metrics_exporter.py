from prometheus_client import start_http_server, Gauge
import psutil, time

# Define metrics
cpu_usage = Gauge("system_cpu_percent", "CPU utilization percentage")
mem_usage = Gauge("system_memory_percent", "Memory utilization percentage")

if __name__ == "__main__":
    print("Starting local metrics exporter on port 8000...")
    start_http_server(8000)  # Prometheus will scrape here
    while True:
        cpu_usage.set(psutil.cpu_percent(interval=None))
        mem_usage.set(psutil.virtual_memory().percent)
        time.sleep(1)
