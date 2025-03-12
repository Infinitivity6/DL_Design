# services/dataFill.py

import os
import pandas as pd
from flask import current_app

def get_missing_details(dataset, current_page, page_size):
    """
    如果 dataset 不为 None, 就只返回该数据集的缺失信息;
    否则返回 train/validation/test 三个的数据。
    这里演示一个简化写法:
      - 优先读 preProcess/<ds>.csv, 若无则读 uploads/<ds>.csv
      - 找出所有缺失值 (rowIndex, colName)
      - 分页后返回
    """
    ds_list = ['train','validation','test'] if not dataset else [dataset]
    result = {}
    for ds in ds_list:
        info = _get_single_dataset_missing(ds, current_page, page_size)
        result[ds] = info
    return result

def _get_single_dataset_missing(ds, current_page, page_size):
    preproc_folder = current_app.config['PREPROCESSED_FOLDER']
    upload_folder = current_app.config['UPLOAD_FOLDER']

    preproc_path = os.path.join(preproc_folder, f"{ds}.csv")
    upload_path = os.path.join(upload_folder, f"{ds}.csv")

    if not os.path.exists(preproc_path) and not os.path.exists(upload_path):
        return {"message": f"{ds} 数据集尚未上传"}

    # 读最新版本
    src_path = preproc_path if os.path.exists(preproc_path) else upload_path
    try:
        df = pd.read_csv(src_path)
    except:
        return {"message": f"读取 {ds} 数据集失败"}

    # 找缺失值
    missing_cells = []
    for col in df.columns:
        col_na_indices = df[df[col].isna()].index.tolist()
        for row_idx in col_na_indices:
            missing_cells.append({
                "row": int(row_idx),
                "colName": col
            })

    total_missing = len(missing_cells)
    if total_missing == 0:
        return {"message": f"{ds} 数据集没有缺失值"}

    # 分页
    start_idx = (current_page - 1)*page_size
    end_idx = start_idx + page_size
    page_cells = missing_cells[start_idx:end_idx]

    has_more = (end_idx < total_missing)

    return {
      "missingCells": page_cells,
      "totalMissingCount": total_missing,
      "hasMore": has_more,
      "currentPage": current_page
    }


def apply_fill_instructions(instructions):
    """
    instructions形如:
    {
      "train": {
        "fillMode": "manual"/"auto"/"delete",
        "cells": [
          {"row":12, "colName":"A", "method":"mean"/"interpolation"/"specific", "fillValue":...},
          ...
        ]
      },
      "validation": {...},
      "test": {...}
    }
    如果 fillMode="delete", 表示删除所有缺失值行
    如果 fillMode="auto", 表示一键自动填充
    如果 fillMode="manual", 表示对 cells 逐条处理
    """
    results = {}
    for ds, detail in instructions.items():
        fill_mode = detail.get('fillMode','manual')
        preproc_folder = current_app.config['PREPROCESSED_FOLDER']
        upload_folder = current_app.config['UPLOAD_FOLDER']
        preproc_path = os.path.join(preproc_folder, f"{ds}.csv")
        upload_path = os.path.join(upload_folder, f"{ds}.csv")

        if not os.path.exists(preproc_path) and not os.path.exists(upload_path):
            results[ds] = {"message": f"{ds} 数据集尚未上传，无法填充"}
            continue

        src_path = preproc_path if os.path.exists(preproc_path) else upload_path
        try:
            df = pd.read_csv(src_path)
        except:
            results[ds] = {"error": f"读取 {ds} 数据集失败"}
            continue

        if fill_mode == 'delete':
            # 直接删除所有含缺失值的行
            before_len = len(df)
            df.dropna(inplace=True)
            after_len = len(df)
            removed_rows = before_len - after_len
            df.to_csv(preproc_path, index=False)
            results[ds] = {"status":"ok","deletedRows": removed_rows}
            continue

        if fill_mode == 'auto':
            # 后端自动决定: 对数值列用均值, 对文本列用特定值"未知"...
            filled_count = _auto_fill(df)
            df.to_csv(preproc_path, index=False)
            results[ds] = {"status":"ok","autoFilled": filled_count}
            continue

        # 否则 fill_mode=="manual"
        filled_count = 0
        cells = detail.get('cells', [])
        for cell in cells:
            row = cell['row']
            col = cell['colName']
            method = cell['method']
            fill_val = cell.get('fillValue', None)

            # 如果 df[col][row] 不是 NaN, 跳过
            if row >= len(df) or col not in df.columns:
                continue
            if pd.notna(df.at[row,col]):
                continue

            if method == 'mean':
                # 均值
                mean_val = df[col].mean()
                df.at[row,col] = mean_val
                filled_count += 1
            elif method == 'interpolation':
                # 简单做法: df[col].interpolate()
                # 但这里仅针对单个 cell => 不好处理. 需全列插值.
                df[col] = df[col].interpolate(method='linear')
                # 这会填充整列, 可能影响多个 missing cell
                # 计数可再扫描
            elif method == 'specific':
                df.at[row,col] = fill_val
                filled_count += 1
            # 其他方法可扩展...

        df.to_csv(preproc_path, index=False)
        results[ds] = {"status":"ok","filledCount": filled_count}

    return results


def _auto_fill(df):
    """
    对数值列用均值, 对字符串列用"未知", 返回填充了多少cell
    """
    filled_count = 0
    for col in df.columns:
        na_indices = df[df[col].isna()].index
        if len(na_indices)==0:
            continue
        # 判断列是否数值
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            filled_count += len(na_indices)
        else:
            df[col].fillna("未知", inplace=True)
            filled_count += len(na_indices)
    return filled_count
