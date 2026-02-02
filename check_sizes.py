import os

def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

root = r"c:\Users\seungsu\madcamp_2025_winter_week4"
print(f"Checking sizes in {root}...")
for item in os.listdir(root):
    path = os.path.join(root, item)
    if os.path.isdir(path):
        size = get_dir_size(path)
        print(f"{item}: {format_size(size)}")
    else:
        size = os.path.getsize(path)
        print(f"{item}: {format_size(size)}")
