#!/bin/bash

# 定义要传递的参数列表
params=(
    "16"
    "32"
    "64"
    "128"
)

# 循环遍历参数列表
for param in "${params[@]}"
do
    python main.py --output-dim "$param"
done