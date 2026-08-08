#! /usr/bin/env python3
import asyncio


def test_basic_query():
    from theoros.theoros_main import basic_movie_query

    print("Testing basic movie query...")
    query = "Who directed the movie Inception?"
    result = asyncio.run(basic_movie_query(query))
    assert "Christopher Nolan" in result

def test_graph_like_query():
    from theoros.theoros_main import graph_like_query

    print("Testing graph-like movie query...")
    query = "List all movies directed by Christopher Nolan."
    result = asyncio.run(graph_like_query(query))
    assert "Inception" in result and "Interstellar" in result and "Dunkirk" in result

def test_plot_query():
    from theoros.theoros_main import plot_query

    print("Testing plot query...")
    query = "Recommend a movie based on the plot of Inception."
    result = asyncio.run(plot_query(query))
    assert "dream" in result.lower() or "mind-bending" in result.lower()

def test_pitch_query():
    from theoros.theoros_main import pitch_query

    print("Testing pitch query...")
    query = "Generate a movie pitch based on the following bullet points: [hero, adventure, young]"
    result = asyncio.run(pitch_query(query))
    assert "hero" in result.lower() and "adventure" in result.lower()

def test_pitch_review():
    from theoros.theoros_main import pitch_review

    print("Testing pitch review...")
    query = "Review the following movie pitch: [A young hero embarks on a thrilling adventure.]"
    result = asyncio.run(pitch_review(query))
    assert "engaging" in result.lower() or "character development" in result.lower()

def main():
    tests = [
        test_basic_query,
        test_graph_like_query,
        test_plot_query,
        test_pitch_query,
        test_pitch_review
    ]
    print("Running smoke tests...")
    for test in tests:
        try:
            test()
            print(f"{test.__name__} passed.")
        except AssertionError:
            print(f"{test.__name__} failed.")
            raise
    print("All tests passed.")

if __name__ == '__main__':
    main()