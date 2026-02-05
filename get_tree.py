import os

def print_tree(startpath):
    print(f"Project Root: {os.path.abspath(startpath)}")
    for root, dirs, files in os.walk(startpath):
        # 过滤掉不需要的文件夹
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * (level)
        print(f'{indent}├── {os.path.basename(root)}/')
        subindent = '│   ' * (level + 1)
        for f in files:
            if not f.startswith('.'):
                print(f'{subindent}├── {f}')

if __name__ == "__main__":
    print_tree('.')