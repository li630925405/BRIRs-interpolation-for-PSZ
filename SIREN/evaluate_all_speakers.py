#!/usr/bin/env python3

import os
import sys
import subprocess

def modify_src_idx(idx):
    """修改setting.py中的src_idx值"""
    with open('setting.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到src_idx行并修改
    for i, line in enumerate(lines):
        if line.startswith('src_idx = '):
            lines[i] = f'src_idx = {idx}\n'
            break
    
    with open('setting.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)

def modify_num_x(num_x_val):
    """修改setting.py中的num_x值"""
    with open('setting.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到num_x行并修改
    for i, line in enumerate(lines):
        if line.startswith('num_x = '):
            lines[i] = f'num_x = {num_x_val}\n'
            break
    
    with open('setting.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)

def modify_data_type(data_type_val):
    """修改setting.py中的data_type值"""
    with open('setting.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到data_type行并修改
    for i, line in enumerate(lines):
        if line.startswith('data_type = '):
            lines[i] = f'data_type = "{data_type_val}"\n'
            break
    
    with open('setting.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)

def main():
    num_x_values = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111]
    data_type_values = ["brir", "brir_r"]
    print(f"开始自动评估所有数据类型、扬声器和麦克风配置")
    print(f"数据类型: {data_type_values}")
    print(f"麦克风数量: {num_x_values}")
    print(f"扬声器索引: 0-5")
    
    results = []
    
    for data_type_val in data_type_values:
        print(f"\n{'='*80}")
        print(f"正在评估数据类型: {data_type_val}")
        print(f"{'='*80}")
        
        # 修改data_type
        modify_data_type(data_type_val)
        print(f"已将data_type设置为 {data_type_val}")
        
        for num_x_val in num_x_values:
            print(f"\n{'='*60}")
            print(f"正在评估 {num_x_val} 个麦克风配置 (data_type = {data_type_val})")
            print(f"{'='*60}")
            
            # 修改num_x
            modify_num_x(num_x_val)
            print(f"已将num_x设置为 {num_x_val}")
            
            for idx in range(6):  # 0到5共6个扬声器
                print(f"\n{'-'*40}")
                print(f"正在评估扬声器 {idx} (data_type = {data_type_val}, num_x = {num_x_val}, src_idx = {idx})")
                print(f"{'-'*40}")
                
                # 修改src_idx
                modify_src_idx(idx)
                print(f"已将src_idx设置为 {idx}")
                
                # 运行评估
                try:
                    result = subprocess.run([sys.executable, 'evaluate.py'], 
                                          capture_output=False, 
                                          text=True, 
                                          check=True)
                    print(f"扬声器 {idx} (data_type = {data_type_val}, num_x = {num_x_val}) 评估完成!")
                except subprocess.CalledProcessError as e:
                    print(f"扬声器 {idx} (data_type = {data_type_val}, num_x = {num_x_val}) 评估失败: {e}")
                    print("是否继续评估下一个扬声器? (y/n): ", end='')
                    choice = input().lower()
                    if choice != 'y':
                        print("评估中断")
                        return
    
    print(f"\n{'='*80}")
    print("所有数据类型、扬声器和麦克风配置评估完成!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()