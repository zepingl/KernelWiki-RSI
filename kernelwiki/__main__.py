"""Local knowledge CLI; source bundles continue to use KernelClient."""

import argparse
import json

from .knowledge import EvidenceRef, KernelState, KnowledgePolicy, KnowledgeRecord
from .knowledge_store import KnowledgeClient, KnowledgeUpdate


def main():
    parser = argparse.ArgumentParser(description="Query and version KernelWiki knowledge")
    parser.add_argument("--root", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--revision")
    query = commands.add_parser("query")
    query.add_argument("--state", required=True, help="JSON KernelState")
    query.add_argument("--revision")
    query.add_argument("--top-k", type=int, default=3)
    publish = commands.add_parser("publish")
    publish.add_argument("--candidate", required=True)
    publish.add_argument("--expected-revision", required=True)
    rollback = commands.add_parser("rollback")
    rollback.add_argument("--revision", required=True)
    rollback.add_argument("--expected-revision", required=True)
    commands.add_parser("recover")
    args = parser.parse_args()
    client = KnowledgeClient(args.root)
    if args.command == "query":
        result = [record.to_dict() for record in client.query(KernelState.from_dict(json.loads(args.state)),
                                                             args.revision, args.top_k)]
    elif args.command == "snapshot":
        result = client.snapshot(args.revision)
        result = {"revision": result.revision, "policy": result.policy.to_dict(),
                  "records": [record.to_dict() for record in result.records]}
    elif args.command == "publish":
        with open(args.candidate) as stream:
            value = json.load(stream)
        candidate = KnowledgeUpdate(KnowledgePolicy.from_dict(value["policy"]),
                                    tuple(KnowledgeRecord.from_dict(row) for row in value["records"]),
                                    value["validation"], tuple(EvidenceRef.from_dict(row) for row in value["evidence"]))
        result = {"revision": client.publish(candidate, args.expected_revision)}
    elif args.command == "rollback":
        result = {"revision": client.rollback(args.revision, args.expected_revision)}
    else:
        result = {"revision": client.recover()}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
