# routes/dataFill_routes.py
# 该文件的主要作用是响应前端对于数据预处理_数据填充的请求

import os
from flask import Blueprint, request, jsonify
from services.dataFill import get_missing_details, apply_fill_instructions

data_fill_bp = Blueprint('data_fill_bp', __name__)

@data_fill_bp.route('/missing_details', methods=['GET'])
def missing_details():
    """
    前端请求 /api/dataPreprocess/missing_details?dataset=train&currentPage=1&pageSize=9
    或者不带 dataset 表示一次性获取 train/validation/test 三个的缺失信息
    返回 { train: {...}, validation: {...}, test: {...} }
    每个数据集下: { "missingCells": [...], "totalMissingCount":..., "hasMore":... }
    也可以返回 "message" if not uploaded, or "no missing" if no missing
    """
    dataset = request.args.get('dataset', None)
    current_page = int(request.args.get('currentPage', 1))
    page_size = int(request.args.get('pageSize', 9))

    result = get_missing_details(dataset, current_page, page_size)
    return jsonify(result), 200

@data_fill_bp.route('/apply_fill', methods=['POST'])
def apply_fill():
    """
    接收前端提交的缺失值填充指令:
    {
      "instructions": {
         "train": {
            "fillMode": "manual", // 或 "auto" / "delete"
            "cells": [
               {"row":12,"colName":"A","method":"mean","fillValue":null},
               {"row":15,"colName":"B","method":"specific","fillValue":"未知"},
               ...
            ]
         },
         "validation": {...},
         "test": {...}
      }
    }
    如果 fillMode="auto" 表示一键填充(后端自己判断数值列/文本列?)
    如果 fillMode="delete" 表示删除所有缺失值行
    返回 { "status":"success", "details": "...", "filledCount":..., "deletedRows":... }
    """
    data = request.json
    instructions = data.get('instructions', {})
    result = apply_fill_instructions(instructions)
    return jsonify(result), 200
