"""Unit tests for speaker matching."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from speaker.matcher import (
    compute_distance,
    find_best_match,
    is_clear_winner,
    match_clusters,
    merge_matched_clusters
)


def make_embedding(seed=0):
    """Create a random 192-dim embedding."""
    np.random.seed(seed)
    emb = np.random.randn(192).astype(float)
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


def make_voiceprint(name, seed=0):
    """Create a mock voiceprint profile."""
    return {
        "name": name,
        "embedding": make_embedding(seed),
        "pitch_hz": 120 + seed * 5,
        "energy_rms": 0.05 + seed * 0.01,
        "total_speech_sec": 60.0
    }


class TestComputeDistance:
    def test_returns_dict_with_all_distances(self):
        cluster_emb = make_embedding(0)
        voiceprint = make_voiceprint("Test", 0)
        
        result = compute_distance(cluster_emb, 120, 0.05, voiceprint)
        
        assert "emb_dist" in result
        assert "pitch_dist" in result
        assert "energy_dist" in result
        assert "combined" in result
        assert "confidence" in result
    
    def test_same_embedding_gives_zero_distance(self):
        emb = make_embedding(0)
        voiceprint = {"embedding": emb, "pitch_hz": 120, "energy_rms": 0.05}
        
        result = compute_distance(emb, 120, 0.05, voiceprint)
        
        assert result["emb_dist"] == 0.0
    
    def test_different_embeddings_give_positive_distance(self):
        cluster_emb = make_embedding(0)
        voiceprint = make_voiceprint("Test", 1)
        
        result = compute_distance(cluster_emb, 120, 0.05, voiceprint)
        
        assert result["emb_dist"] > 0


class TestFindBestMatch:
    def test_finds_correct_match(self):
        cluster_emb = make_embedding(0)
        voiceprints = {
            "SpeakerA": make_voiceprint("SpeakerA", 0),
            "SpeakerB": make_voiceprint("SpeakerB", 1),
        }
        
        best_name, best_dist, best_conf, distances = find_best_match(
            cluster_emb, 120, 0.05, voiceprints
        )
        
        assert best_name == "SpeakerA"
        assert best_dist < 0.3
    
    def test_returns_empty_on_no_voiceprints(self):
        cluster_emb = make_embedding(0)
        
        best_name, best_dist, best_conf, distances = find_best_match(
            cluster_emb, 120, 0.05, {}
        )
        
        assert best_name is None
        assert best_dist == float('inf')
        assert distances == {}


class TestIsClearWinner:
    def test_clear_winner_when_gap_large(self):
        matches = [
            ("SpeakerA", 0.1, 0.9),
            ("SpeakerB", 0.3, 0.7),
        ]
        voiceprints = {
            "SpeakerA": {"total_speech_sec": 60},
            "SpeakerB": {"total_speech_sec": 60},
        }
        
        result = is_clear_winner(matches, voiceprints)
        
        assert result is True
    
    def test_not_clear_winner_when_gap_small(self):
        matches = [
            ("SpeakerA", 0.2, 0.8),
            ("SpeakerB", 0.21, 0.79),
        ]
        voiceprints = {
            "SpeakerA": {"total_speech_sec": 60},
            "SpeakerB": {"total_speech_sec": 60},
        }
        
        result = is_clear_winner(matches, voiceprints)
        
        assert result is False
    
    def test_clear_winner_with_more_training_data(self):
        matches = [
            ("SpeakerA", 0.04, 0.96),
            ("SpeakerB", 0.05, 0.95),
        ]
        voiceprints = {
            "SpeakerA": {"total_speech_sec": 600},
            "SpeakerB": {"total_speech_sec": 60},
        }
        
        result = is_clear_winner(matches, voiceprints)
        
        assert result is True


class TestMatchClusters:
    def test_matches_clusters_to_voiceprints(self):
        clusters = {
            "SPEAKER0": {"embedding": make_embedding(0), "pitch_hz": 120, "energy_rms": 0.05},
            "SPEAKER1": {"embedding": make_embedding(1), "pitch_hz": 130, "energy_rms": 0.06},
        }
        voiceprints = {
            "SpeakerA": make_voiceprint("SpeakerA", 0),
            "SpeakerB": make_voiceprint("SpeakerB", 1),
        }
        
        results = match_clusters(clusters, voiceprints)
        
        assert "SPEAKER0" in results
        assert results["SPEAKER0"]["name"] == "SpeakerA"
        assert results["SPEAKER1"]["name"] == "SpeakerB"
    
    def test_no_match_when_distance_too_high(self):
        clusters = {
            "SPEAKER0": {"embedding": make_embedding(0), "pitch_hz": 120, "energy_rms": 0.05},
        }
        voiceprints = {
            "SpeakerA": make_voiceprint("SpeakerA", 99),  # Very different embedding
        }
        
        results = match_clusters(clusters, voiceprints)
        
        assert results["SPEAKER0"]["matched"] is False


class TestMergeMatchedClusters:
    def test_merges_clusters_matched_to_same_speaker(self):
        results = {
            "SPEAKER0": {"name": "SpeakerA", "matched": True},
            "SPEAKER1": {"name": "SpeakerA", "matched": True},
            "SPEAKER2": {"name": "SpeakerB", "matched": True},
        }
        
        final_map = merge_matched_clusters(results, {})
        
        # Check that matched clusters get names
        assert final_map["SPEAKER0"] == "SpeakerA"
        assert final_map["SPEAKER1"] == "SpeakerA"
        assert final_map["SPEAKER2"] == "SpeakerB"
    
    def test_unmatched_clusters_get_speaker_labels(self):
        results = {
            "SPEAKER0": {"name": None, "matched": False},
            "SPEAKER1": {"name": None, "matched": False},
        }
        
        final_map = merge_matched_clusters(results, {})
        
        assert final_map["SPEAKER0"].startswith("SPEAKER")
        assert final_map["SPEAKER1"].startswith("SPEAKER")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])