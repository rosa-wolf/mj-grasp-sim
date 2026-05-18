import os

base_dir = "/home/ws/data/outputs/context_clutter_v3/PandaGripper"

deleted = 0
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "scene_pcd.npz":
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
print(f"Total deleted: {deleted}")
