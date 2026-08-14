"""Basic unit tests for text cleaning and feature extraction. Run with: pytest tests/"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import clean_text
from src.models.skill_gap import extract_skills
from src.features.match_features import (
    skill_overlap_features, education_match_feature, experience_required_years,
)


def test_clean_text_lowercases_and_strips_html():
    assert clean_text("<b>Hello WORLD</b>") == "hello world"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_extract_skills_finds_known_terms():
    text = clean_text("Experienced in python, sql, and project management.")
    skills = extract_skills(text)
    assert "python" in skills
    assert "sql" in skills
    assert "project management" in skills


def test_extract_skills_empty_for_no_matches():
    text = clean_text("A completely unrelated sentence about weather.")
    assert extract_skills(text) == set()


def test_skill_overlap_features_jaccard():
    resume = clean_text("python sql excel")
    job = clean_text("python excel tableau")
    feats = skill_overlap_features(resume, job)
    assert feats["skill_overlap_count"] == 2  # python, excel
    assert 0 < feats["skill_jaccard"] < 1


def test_education_match_feature():
    resume = clean_text("Bachelor of Science in Computer Science")
    job_with_degree = clean_text("Requires a bachelor's degree")
    job_without_degree = clean_text("No formal education required")
    assert education_match_feature(resume, job_with_degree) == 1
    assert education_match_feature(resume, job_without_degree) == 0


def test_experience_required_years_parses_range():
    assert experience_required_years("Work Experience 2 - 5 Years") == 3.5


def test_experience_required_years_handles_missing():
    assert experience_required_years("no experience info here") == 0.0
    assert experience_required_years(None) == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
