#!/usr/bin/env python3
"""
DataMind AI — CLI
Usage:
  python run.py --file data.csv --target price
  python run.py --demo
"""

import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="DataMind AI — Council of Agents")
    parser.add_argument("--file", "-f")
    parser.add_argument("--target", "-t")
    parser.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    parser.add_argument("--groq-key", default=os.getenv("GROQ_API_KEY", ""))
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        from sklearn.datasets import load_iris
        import pandas as pd
        df = load_iris(as_frame=True).frame
        df.to_csv("/tmp/iris_demo.csv", index=False)
        args.file = "/tmp/iris_demo.csv"
        args.target = "target"
        print("🌸 Running demo on Iris dataset\n")
    elif not args.file or not args.target:
        parser.print_help(); sys.exit(1)

    import pandas as pd
    if args.file.endswith(".csv"):
        df = pd.read_csv(args.file)
    elif args.file.endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.file)
    else:
        df = pd.read_parquet(args.file)

    task_type = args.task
    if task_type == "auto":
        tgt = df[args.target]
        task_type = "classification" if (tgt.dtype == "object" or tgt.nunique() <= 20) else "regression"

    from agents.orchestrator import Orchestrator, AgentMemory
    from agents.council import EDAAgent, MLAgent, InsightAgent, CriticAgent, CodeAgent
    from agents.synthesis import SynthesisAgent

    def cb(name, emoji, msg):
        print(f"  {emoji} [{name}] {msg}")

    memory = AgentMemory()
    orch = Orchestrator(progress_callback=cb)
    for Cls in [EDAAgent, MLAgent, InsightAgent, CriticAgent, CodeAgent]:
        orch.register_agent(Cls(memory=memory, progress_callback=cb))
    orch.register_agent(SynthesisAgent(memory=memory, groq_api_key=args.groq_key, progress_callback=cb))

    print("=" * 60)
    print("  🧠 DataMind AI — Council of Agents")
    print("=" * 60)

    result = orch.run(df, args.target, task_type)
    findings = result["findings"]

    synth = findings.get("Synthesis", {})
    print("\n" + "=" * 60)
    print("  📄 REPORT")
    print("=" * 60)
    print(synth.get("narrative", "No report generated."))

    max_f = findings.get("Max", {})
    print(f"\n🏆 Best model: {max_f.get('best_model')} ({max_f.get('metric_name')}={max_f.get('best_score', 0):.4f})")
    print(f"⏱️  Completed in {result['elapsed_seconds']}s")

    with open("datamind_report.md", "w") as f:
        f.write(synth.get("narrative", ""))
    print("\n📄 Report saved: datamind_report.md")


if __name__ == "__main__":
    main()
