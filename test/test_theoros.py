import asyncio
import unittest

from pydantic_ai import capture_run_messages
from pydantic_ai.models.test import TestModel


class TestTheoros(unittest.TestCase):
    def test_movie_expert(self):
        from theoros.movie_expert import movie_expert
        from theoros.theoros_main import basic_movie_query
        expected = "Christopher Nolan"
        def mock_search(query):
            return expected
        with capture_run_messages() as captured:
            with movie_expert.override(
                    model=TestModel(custom_output_text=expected),
                    tools=[mock_search]):
                query = "Who directed the movie Inception?"
                result = asyncio.run(basic_movie_query(query))
                self.assertTrue(expected in captured[-1].parts[0].content)

    def test_graph_like_query(self):
        from theoros.graph_like_search import graph_search_agent
        from theoros.theoros_main import graph_like_query
        expected = "wd:Q25191"  # Q-ID for Christopher
        def mock_build_sparql(query):
            return "SELECT ?director WHERE { wd:Q25188 wdt:P57 ?director }"
        def mock_run_sparql(query):
            return {"results": {"bindings": [{"director": {
                "type": "uri",
                "value": "http://www.wikidata.org/entity/Q25188"
            }}]}}
        with capture_run_messages() as captured:
            with graph_search_agent.override(
                    model=TestModel(custom_output_text=expected),
                    tools=[mock_build_sparql, mock_run_sparql]) :
                query = "Who directed the movie Inception?"
                result = asyncio.run(graph_like_query(query))
                self.assertTrue(expected in captured[-1].parts[0].content)

    def test_plot_query(self):
        from theoros.plot_search import plot_search_agent
        from theoros.theoros_main import plot_query
        expected = "A mind-bending thriller about dreams within dreams."
        def mock_plot_search(query):
            return expected
        with capture_run_messages() as captured:
            with plot_search_agent.override(
                    model=TestModel(custom_output_text=expected),
                    tools=[mock_plot_search]):
                query = "Recommend a movie based on the plot of Inception."
                result = asyncio.run(plot_query(query))
                self.assertTrue(expected in captured[-1].parts[0].content)

    def test_pitch_query(self):
        from theoros.pitch_builder import pitch_agent
        from theoros.theoros_main import pitch_query
        expected = "A thrilling adventure of a young hero."
        def mock_pitch_builder(query):
            return expected
        with capture_run_messages() as captured:
            with pitch_agent.override(
                    model=TestModel(custom_output_text=expected),
                    tools=[mock_pitch_builder]):
                query = ("Generate a movie pitch based on the following bullet "
                         "points: [hero, adventure, young]")
                result = asyncio.run(pitch_query(query))
                self.assertEqual(expected, captured[-1].parts[0].content)

    def test_pitch_review(self):
        from theoros.pitch_reviewer import pitch_review_agent
        from theoros.theoros_main import pitch_review
        expected = ("The plot is engaging, but consider adding more character "
                    "development.")
        def mock_pitch_reviewer(query):
            return expected
        with capture_run_messages() as captured:
            with pitch_review_agent.override(
                    model=TestModel(custom_output_text=expected),
                    tools=[mock_pitch_reviewer]):
                query = ("Review the following movie pitch: [A young hero embarks on "
                         "a thrilling adventure.]")
                result = asyncio.run(pitch_review(query))
                self.assertEqual(expected, captured[-1].parts[0].content)


if __name__ == '__main__':
    unittest.main()
