import json
import re

from theoros.theoros_main import theoros_agent

async def main():
    data = []
    with open("./test_data.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    responses = []
    for item in data:
        query = item["prompt"]
        expected_answer = item["expected_answer"]

        actual_answer = await theoros_agent.run(query)
        actual_answer = actual_answer.response.text.strip()

        response = dict(**item)
        response["actual_answer"] = actual_answer
        if item["relation"] in ("directed", "screenwriter", "stars-in"):
            response["match"] = all(ea.lower() in actual_answer.lower() for ea in expected_answer)
        elif item["relation"] in ("release-year",):
            response["match"] = str(expected_answer) in actual_answer
        elif item["relation"] in ("genre",):
            response["match"] = any(ea.lower() in actual_answer.lower() for ea in expected_answer)
        elif item["relation"] in ("studio",):
            response["match"] = re.findall(expected_answer, actual_answer, re.IGNORECASE) != []

        responses.append(response)
        print(f"Query: {query}\nExpected: {expected_answer}\nActual: {actual_answer}\nMatch: {response['match']}\n")

    with open("./movie_facts_test_results.json", "w") as f:
        json.dump(responses, f, indent=4)

    mean_correct = sum([1 for r in responses if r["match"]]) / len(responses)
    print(f"Mean correct: {mean_correct:.2%}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())