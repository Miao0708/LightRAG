#!/usr/bin/env python3
"""
LightRAG 简单控制脚本
用于暂停/恢复流水线和重置数据库

使用方法:
python lightrag_control.py pause              # 暂停流水线
python lightrag_control.py resume             # 恢复流水线
python lightrag_control.py stop               # 强制停止流水线
python lightrag_control.py reset              # 重置所有数据库
python lightrag_control.py status             # 查看状态
"""

import argparse
import requests
import json
import time
from datetime import datetime

class LightRAGController:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        
    def _post(self, endpoint, data=None):
        """发送POST请求"""
        try:
            url = f"{self.api_url}{endpoint}"
            response = requests.post(url, json=data, timeout=30)
            return response.json() if response.content else {"status": "success"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"连接失败: {str(e)}"}
    
    def _get(self, endpoint):
        """发送GET请求"""
        try:
            url = f"{self.api_url}{endpoint}"
            response = requests.get(url, timeout=10)
            return response.json() if response.content else {"status": "success"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"连接失败: {str(e)}"}
    
    def pause_pipeline(self):
        """暂停流水线"""
        print("🟡 正在暂停流水线...")
        result = self._post("/documents/pipeline/pause")
        
        if result.get("status") == "success":
            print("✅ 流水线暂停成功")
        else:
            print(f"❌ 暂停失败: {result.get('message', '未知错误')}")
        return result
    
    def resume_pipeline(self):
        """恢复流水线"""
        print("🟢 正在恢复流水线...")
        result = self._post("/documents/pipeline/resume")
        
        if result.get("status") == "success":
            print("✅ 流水线恢复成功")
        else:
            print(f"❌ 恢复失败: {result.get('message', '未知错误')}")
        return result
    
    def force_stop_pipeline(self):
        """强制停止流水线"""
        print("🔴 正在强制停止流水线...")
        result = self._post("/documents/pipeline/force_stop")
        
        if result.get("status") == "success":
            print("✅ 流水线强制停止成功")
        else:
            print(f"❌ 强制停止失败: {result.get('message', '未知错误')}")
        return result
    
    def reset_databases(self, confirm=False):
        """重置所有数据库"""
        if not confirm:
            print("⚠️  警告: 此操作将删除所有数据库内容！")
            print("如需执行，请运行: python lightrag_control.py reset --confirm")
            return {"status": "cancelled", "message": "用户取消操作"}
        
        print("🔥 正在重置所有数据库...")
        print("⏳ 这可能需要一些时间...")
        
        result = self._post("/documents/reset/all_databases")
        
        if result.get("status") == "success":
            print("✅ 数据库重置完成")
            details = result.get("details", {})
            print("重置详情:")
            for component, status in details.items():
                emoji = "✅" if status == "成功" else "❌"
                print(f"  {emoji} {component}: {status}")
        else:
            print(f"❌ 重置失败: {result.get('message', '未知错误')}")
        
        return result
    
    def get_status(self):
        """获取流水线状态"""
        result = self._get("/documents/pipeline_status")
        
        if "error" in result:
            print(f"❌ 获取状态失败: {result['message']}")
            return result
        
        print("📊 LightRAG 流水线状态")
        print("=" * 40)
        
        # 基本状态
        busy = result.get("busy", False)
        paused = result.get("paused", False)
        
        if paused:
            status_text = "🟡 已暂停"
        elif busy:
            status_text = "🟢 运行中"
        else:
            status_text = "⚪ 空闲"
        
        print(f"状态: {status_text}")
        print(f"任务: {result.get('job_name', '无')}")
        
        # 进度信息
        if busy and not paused:
            cur_batch = result.get("cur_batch", 0)
            total_batch = result.get("batchs", 0)
            if total_batch > 0:
                progress = (cur_batch / total_batch) * 100
                print(f"进度: {cur_batch}/{total_batch} ({progress:.1f}%)")
        
        # 时间信息
        if result.get("job_start"):
            print(f"开始时间: {result['job_start']}")
        
        # 最新消息
        if result.get("latest_message"):
            print(f"最新消息: {result['latest_message']}")
        
        return result
    
    def monitor(self, interval=5):
        """监控流水线状态"""
        print("🔍 开始监控流水线状态...")
        print("按 Ctrl+C 停止监控")
        
        try:
            while True:
                print(f"\n--- {datetime.now().strftime('%H:%M:%S')} ---")
                status = self.get_status()
                
                if not status.get("busy", False):
                    print("✅ 流水线已完成所有任务")
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 监控已停止")

def main():
    parser = argparse.ArgumentParser(description="LightRAG 流水线控制工具")
    parser.add_argument("action", choices=["pause", "resume", "stop", "reset", "status", "monitor"],
                       help="要执行的操作")
    parser.add_argument("--confirm", action="store_true", 
                       help="确认执行危险操作（用于reset）")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="LightRAG API 地址 (默认: http://localhost:8000)")
    parser.add_argument("--interval", type=int, default=5,
                       help="监控间隔秒数 (默认: 5)")
    
    args = parser.parse_args()
    
    controller = LightRAGController(args.url)
    
    if args.action == "pause":
        controller.pause_pipeline()
    elif args.action == "resume":
        controller.resume_pipeline()
    elif args.action == "stop":
        controller.force_stop_pipeline()
    elif args.action == "reset":
        controller.reset_databases(confirm=args.confirm)
    elif args.action == "status":
        controller.get_status()
    elif args.action == "monitor":
        controller.monitor(interval=args.interval)

if __name__ == "__main__":
    main()