import shutil

total, used, free = shutil.disk_usage("C:/")

print(f"Total: {total / (2**30):.2f} GB")
print(f"Used: {used / (2**30):.2f} GB")
print(f"Free: {free / (2**30):.2f} GB")
