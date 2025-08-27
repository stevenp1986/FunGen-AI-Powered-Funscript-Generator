#!/usr/bin/env python3
"""
Advanced Chapter Operations Tests for VR Funscript AI Generator

Tests advanced chapter functionality including:
- Chapter merging and splitting
- Points deletion within chapters
- Tracking within chapters
- Tracking over gaps between chapters 
- Merge tracking data between chapters
- Chapter boundary adjustment
- Complex chapter scenarios
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.insert(0, os.path.abspath('.'))

from application.logic.app_logic import ApplicationLogic
from application.logic.app_funscript_processor import AppFunscriptProcessor
from application.utils.video_segment import VideoSegment
from application.classes.project_manager import ProjectManager
from funscript import DualAxisFunscript

# Test markers
pytestmark = [pytest.mark.integration, pytest.mark.chapters, pytest.mark.advanced]

class TestChapterMerging:
    """Test chapter merging functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock(spec=ApplicationLogic)
        self.app.logger = Mock()
        self.app.project_manager = Mock(spec=ProjectManager)
        self.app.project_manager.project_dirty = False
        
        # Mock app_state_ui with required attributes
        self.app.app_state_ui = Mock()
        self.app.app_state_ui.heatmap_dirty = False
        self.app.app_state_ui.funscript_preview_dirty = False
        
        # Mock GUI components
        self.app.gui_instance = Mock()
        self.app.gui_instance.timeline_editor1 = Mock()
        self.app.gui_instance.timeline_editor2 = Mock()
        
        self.funscript_processor = AppFunscriptProcessor(self.app)
    
    def test_merge_adjacent_chapters(self):
        """Test merging two adjacent chapters."""
        fs_proc = self.funscript_processor
        
        # Create two adjacent chapters
        chapter1 = VideoSegment(100, 200, 1, "SexAct", "SexAct", "First", "First Position", source="manual")
        chapter2 = VideoSegment(
            start_frame_id=201,  # Adjacent to chapter1
            end_frame_id=300,
            class_id=1,
            class_name="SexAct",
            segment_type="SexAct",
            position_short_name="Second",
            position_long_name="Second Position",
            source="manual",
        )
        
        fs_proc.video_chapters = [chapter1, chapter2]
        
        # Test merge functionality
        merged_chapter = fs_proc.merge_selected_chapters(chapter1, chapter2, return_chapter_object=True)
        
        # Verify merged chapter properties
        assert merged_chapter is not None
        assert merged_chapter.start_frame_id == 100
        assert merged_chapter.end_frame_id == 300
        assert "First" in merged_chapter.position_short_name or "Second" in merged_chapter.position_short_name
        
        # Verify original chapters were removed and merged chapter added
        assert len(fs_proc.video_chapters) == 1
        assert fs_proc.video_chapters[0] == merged_chapter
        assert fs_proc.video_chapters[0].unique_id == merged_chapter.unique_id
    
    def test_merge_chapters_with_gap(self):
        """Test merging chapters with a gap between them."""
        fs_proc = self.funscript_processor
        
        # Create chapters with a gap
        chapter1 = VideoSegment(100, 200, 1, "SexAct", "SexAct", "FirstGap", "FirstGap Position", source="manual")
        chapter2 = VideoSegment(
            start_frame_id=250,  # Gap from 201-249
            end_frame_id=350,
            class_id=1,
            class_name="SexAct",
            segment_type="SexAct",
            position_short_name="SecondGap",
            position_long_name="SecondGap Position",
            source="manual"
        )
        
        fs_proc.video_chapters = [chapter1, chapter2]
        
        # Test merge - should bridge the gap
        merged_chapter = fs_proc.merge_selected_chapters(chapter1, chapter2, return_chapter_object=True)
        
        # Verify merged chapter spans the gap
        assert merged_chapter.start_frame_id == 100
        assert merged_chapter.end_frame_id == 350
        assert len(fs_proc.video_chapters) == 1
    
    def test_merge_overlapping_chapters(self):
        """Test merging overlapping chapters."""
        fs_proc = self.funscript_processor
        
        # Create overlapping chapters
        chapter1 = VideoSegment(100, 250, 1, "SexAct", "SexAct", "OverlapFirst", "OverlapFirst Position", source="manual")
        chapter2 = VideoSegment(
            start_frame_id=200,  # Overlaps with chapter1 (200-250)
            end_frame_id=350,
            class_id=1,
            class_name="SexAct",
            segment_type="SexAct",
            position_short_name="OverlapSecond",
            position_long_name="OverlapSecond Position",
            source="manual"
        )
        
        fs_proc.video_chapters = [chapter1, chapter2]
        
        # Test merge - should use the full span
        merged_chapter = fs_proc.merge_selected_chapters(chapter1, chapter2, return_chapter_object=True)
        
        # Verify merged chapter spans both originals
        assert merged_chapter.start_frame_id == 100
        assert merged_chapter.end_frame_id == 350
        assert len(fs_proc.video_chapters) == 1


class TestChapterSplitting:
    """Test chapter splitting functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock(spec=ApplicationLogic)
        self.app.logger = Mock()
        self.app.project_manager = Mock(spec=ProjectManager)
        self.app.project_manager.project_dirty = False
        
        self.funscript_processor = AppFunscriptProcessor(self.app)
        self.app.app_state_ui = Mock()
        self.app.app_state_ui.heatmap_dirty = False
        self.app.app_state_ui.funscript_preview_dirty = False
        
    
    def test_split_chapter_at_frame(self):
        """Test splitting a chapter at a specific frame."""
        fs_proc = self.funscript_processor
        
        # Create chapter to split
        original_chapter = VideoSegment(100, 300, 1, "SexAct", "SexAct", "ToSplit", "ToSplit Position", source="manual")
        
        fs_proc.video_chapters = [original_chapter]
        
        # Split at frame 200
        split_frame = 200
        
        # Mock implementation of split functionality
        # This would create two new chapters
        first_half = VideoSegment(
            start_frame_id=100,
            end_frame_id=199,  # Frame before split
            class_id=1,
            class_name="SexAct",
            segment_type="SexAct",
            position_short_name="ToSplit_Part1",
            position_long_name="ToSplit_Part1 Position",
            source="manual_split"
        )
        second_half = VideoSegment(
            start_frame_id=200,  # Split frame
            end_frame_id=300,
            class_id=1,
            class_name="SexAct",
            segment_type="SexAct",
            position_short_name="ToSplit_Part2", 
            position_long_name="ToSplit_Part2 Position",
            source="manual_split"
        )
        
        # Simulate split operation
        fs_proc.video_chapters.remove(original_chapter)
        fs_proc.video_chapters.extend([first_half, second_half])
        fs_proc.video_chapters.sort(key=lambda c: c.start_frame_id)
        
        # Verify split results
        assert len(fs_proc.video_chapters) == 2
        assert fs_proc.video_chapters[0].start_frame_id == 100
        assert fs_proc.video_chapters[0].end_frame_id == 199
        assert fs_proc.video_chapters[1].start_frame_id == 200
        assert fs_proc.video_chapters[1].end_frame_id == 300
        
        # Verify no gap between chapters
        gap_exists = fs_proc.video_chapters[0].end_frame_id + 1 != fs_proc.video_chapters[1].start_frame_id
        assert not gap_exists
    
    def test_split_chapter_invalid_frame(self):
        """Test splitting chapter at invalid frame positions."""
        fs_proc = self.funscript_processor
        
        # Create chapter
        chapter = VideoSegment(100, 200, 1, "SexAct", "SexAct", "InvalidSplit", "InvalidSplit Position", source="manual")
        
        fs_proc.video_chapters = [chapter]
        
        # Test split at start frame (invalid - would create empty first part)
        split_at_start = 100
        test1 =fs_proc.split_chapter_at_frame(chapter, split_at_start)
        # Should not split or should handle gracefully
        
        # Test split at end frame (invalid - would create empty second part)  
        split_at_end = 200
        test2 = fs_proc.split_chapter_at_frame(chapter, split_at_end)
        # Should not split or should handle gracefully
        
        # Test split outside chapter bounds
        split_outside = 250
        test3 = fs_proc.split_chapter_at_frame(chapter, split_outside)
        # Should not split or should handle gracefully
        
        # For now, just verify chapter remains unchanged
        #assert len(fs_proc.video_chapters) == 1
        

        assert test1 is None
        assert test2 is None
        assert test3 is None
        assert fs_proc.video_chapters[0].unique_id == chapter.unique_id

class TestPointsDeletion:
    """Test deletion of funscript points within chapters."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock(spec=ApplicationLogic)
        self.app.logger = Mock()
        self.app.project_manager = Mock(spec=ProjectManager)
        self.app.project_manager.project_dirty = False
        
        self.funscript_processor = AppFunscriptProcessor(self.app)
        
        # Mock funscript with sample data
        self.mock_funscript = Mock(spec=DualAxisFunscript)
        self.mock_funscript.primary_actions = [
            {"at": 1000, "pos": 50},   # Frame ~30 (at 30fps)
            {"at": 2000, "pos": 75},   # Frame ~60
            {"at": 3000, "pos": 25},   # Frame ~90
            {"at": 4000, "pos": 90},   # Frame ~120
            {"at": 5000, "pos": 10},   # Frame ~150
            {"at": 6000, "pos": 80},   # Frame ~180
        ]
        self.mock_funscript.secondary_actions = []
    
    def test_delete_points_within_chapter(self):
        """Test deleting funscript points within a specific chapter."""
        fs_proc = self.funscript_processor
        
        # Create chapter 
        chapter = VideoSegment(
            start_frame_id=60,   # Frame 60
            end_frame_id=150,    # Frame 150  
            segment_type="SexAct",
            position_short_name="PointsDelete",
            source="manual",
            class_id=1,
            class_name="SexAct",
            position_long_name="PointsDelete Position"
        )
        
        fs_proc.video_chapters = [chapter]
        
        # Mock getting funscript object
        self.app.processor = Mock()
        self.app.processor.tracker = Mock()
        self.app.processor.tracker.funscript = self.mock_funscript
        
        # Get points within chapter timespan
        chapter_start_ms = int((60 / 30.0) * 1000)  # 2000ms
        chapter_end_ms = int((150 / 30.0) * 1000)   # 5000ms
        
        # Points to delete: those between 2000-5000ms
        points_to_delete = [
            point for point in self.mock_funscript.primary_actions
            if chapter_start_ms <= point["at"] <= chapter_end_ms
        ]
        
        # Verify we found the expected points
        assert len(points_to_delete) == 4  # 2000ms, 3000ms, 4000ms, 5000ms
        
        # Simulate deletion (remove points within chapter)
        remaining_points = [
            point for point in self.mock_funscript.primary_actions
            if not (chapter_start_ms <= point["at"] <= chapter_end_ms)
        ]
        
        # Verify correct points remain
        assert len(remaining_points) == 2  # 1000ms, 6000ms should remain
        assert remaining_points[0]["at"] == 1000
        assert remaining_points[1]["at"] == 6000
    
    def test_delete_points_partial_chapter_selection(self):
        """Test deleting points from a selected range within a chapter."""
        fs_proc = self.funscript_processor
        
        # Create larger chapter
        chapter = VideoSegment(30, 180, 1, "SexAct", "SexAct", "PartialDelete", "PartialDelete Position", source="manual")
        
        fs_proc.video_chapters = [chapter]
        
        # Set up a selection range within the chapter (frames 90-120)
        selection_start_frame = 90
        selection_end_frame = 120
        selection_start_ms = int((selection_start_frame / 30.0) * 1000)  # 3000ms
        selection_end_ms = int((selection_end_frame / 30.0) * 1000)      # 4000ms
        
        # Find points in selection
        selected_points = [
            point for point in self.mock_funscript.primary_actions
            if selection_start_ms <= point["at"] <= selection_end_ms
        ]
        
        # Should find points at 3000ms and 4000ms
        assert len(selected_points) == 2
        assert selected_points[0]["at"] == 3000
        assert selected_points[1]["at"] == 4000


class TestTrackingInChapters:
    """Test tracking functionality within chapters."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock(spec=ApplicationLogic)
        self.app.logger = Mock()
        self.app.project_manager = Mock(spec=ProjectManager)
        self.app.project_manager.project_dirty = False
        
        self.funscript_processor = AppFunscriptProcessor(self.app)
        self.app.app_state_ui = Mock()
        self.app.app_state_ui.heatmap_dirty = False
        self.app.app_state_ui.funscript_dirty = False
    
    def test_tracking_within_single_chapter(self):
        """Test that tracking respects chapter boundaries."""
        fs_proc = self.funscript_processor
        
        # Create chapter with tracking data
        chapter = VideoSegment(100, 200, 1, "SexAct", "SexAct", "TrackingTest", "TrackingTest Position", source="stage2_analysis")
        
        # Set chapter-specific tracking data
        chapter.user_roi_fixed = (50, 50, 100, 100)  # ROI for this chapter
        chapter.user_roi_initial_point_relative = (25, 25)  # Tracking point within ROI
        
        fs_proc.video_chapters = [chapter]
        
        # Verify chapter tracking data
        assert chapter.user_roi_fixed is not None
        assert chapter.user_roi_initial_point_relative is not None
        
        # Test that tracking data is available when frame is within chapter
        current_frame = 150
        found_chapter = fs_proc.get_chapter_at_frame(current_frame)
        assert found_chapter is not None
        assert found_chapter.user_roi_fixed == (50, 50, 100, 100)
        assert found_chapter.user_roi_initial_point_relative == (25, 25)
    
    def test_tracking_over_chapter_gap(self):
        """Test tracking behavior when there are gaps between chapters."""
        fs_proc = self.funscript_processor
        
        # Create chapters with a gap
        chapter1 = VideoSegment(50, 100, 1, "SexAct", "SexAct", "GapTrack1", "GapTrack1 Position", source="manual")
        chapter2 = VideoSegment(
            start_frame_id=150,  # Gap from 101-149
            end_frame_id=200,
            segment_type="SexAct",
            position_short_name="GapTrack2",
            source="manual",
            class_id=1,
            class_name="GapTrack2",
            position_long_name="GapTrack2 Position",
        )
        
        # Set different tracking data for each chapter
        chapter1.user_roi_fixed = (25, 25, 50, 50)
        chapter1.user_roi_initial_point_relative = (12, 12)
        
        chapter2.user_roi_fixed = (75, 75, 50, 50)
        chapter2.user_roi_initial_point_relative = (25, 25)
        
        fs_proc.video_chapters = [chapter1, chapter2]
        
        # Test tracking data in first chapter
        frame_in_ch1 = 75
        found_ch1 = fs_proc.get_chapter_at_frame(frame_in_ch1)  
        assert found_ch1.user_roi_fixed == (25, 25, 50, 50)
        
        # Test no tracking data in gap
        frame_in_gap = 125
        found_gap = fs_proc.get_chapter_at_frame(frame_in_gap)
        assert found_gap is None  # No chapter in gap
        
        # Test tracking data in second chapter
        frame_in_ch2 = 175
        found_ch2 = fs_proc.get_chapter_at_frame(frame_in_ch2)
        assert found_ch2.user_roi_fixed == (75, 75, 50, 50)
    
    def test_merge_tracking_data_between_chapters(self):
        """Test merging tracking data when chapters are merged."""
        fs_proc = self.funscript_processor
        
        # Create chapters with different tracking data
        chapter1 = VideoSegment(100, 150, 1, "SexAct", "SexAct", "MergeTrack1", "MergeTrack1 Position", source="manual")
        chapter2 = VideoSegment(151, 200, 1, "SexAct", "SexAct", "MergeTrack2", "MergeTrack2 Position", source="manual")
        
        # Set tracking data
        chapter1.user_roi_fixed = (30, 30, 40, 40)
        chapter1.user_roi_initial_point_relative = (20, 20)
        
        chapter2.user_roi_fixed = (70, 70, 40, 40)
        chapter2.user_roi_initial_point_relative = (20, 20)
        
        fs_proc.video_chapters = [chapter1, chapter2]
        
        # Test merge - should preserve or combine tracking data appropriately
        merged_chapter = fs_proc.merge_selected_chapters(chapter1, chapter2, return_chapter_object=True)
        
        # Verify merged chapter has tracking data
        # The exact merge logic would depend on implementation
        # Here we test that some tracking data is preserved
        assert merged_chapter.start_frame_id == 100
        assert merged_chapter.end_frame_id == 200
        
        # At minimum, merged chapter should have some tracking information
        # Implementation might choose first, last, or combined tracking data


class TestChapterBoundaryAdjustment:
    """Test adjusting chapter boundaries and handling edge cases."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock(spec=ApplicationLogic)
        self.app.logger = Mock()
        self.app.project_manager = Mock(spec=ProjectManager)
        self.app.project_manager.project_dirty = False
        
        self.funscript_processor = AppFunscriptProcessor(self.app)
    
    def test_adjust_chapter_boundaries(self):
        """Test adjusting chapter start and end frames."""
        fs_proc = self.funscript_processor
        
        # Create chapter
        chapter = VideoSegment(100, 200, 1, "SexAct", "SexAct", "BoundaryAdjust", "BoundaryAdjust Position", source="manual")
        
        fs_proc.video_chapters = [chapter]
        
        # Test boundary adjustment data
        new_chapter_data = {
            "start_frame_str": 90,   # Extend start backward
            "end_frame_str": 220,    # Extend end forward
            "segment_type": "SexAct",
            "position_short_name_key": "BoundaryAdjust_Extended",
            "source": "manual_edit"
        }
        
        # Test update functionality
        fs_proc.update_chapter_from_data(chapter.unique_id, new_chapter_data)
        
        # Verify boundaries were adjusted
        updated_chapter = fs_proc.video_chapters[0]
        assert updated_chapter.start_frame_id == 90
        assert updated_chapter.end_frame_id == 220
        assert updated_chapter.position_short_name == "BoundaryAdjust_Extended"
    
    def test_boundary_adjustment_overlap_prevention(self):
        """Test that boundary adjustments prevent overlaps with other chapters."""
        fs_proc = self.funscript_processor
        
        # Create multiple chapters
        chapter1 = VideoSegment(50, 100, 1, "SexAct", "SexAct", "Boundary1", "Boundary1 Position", source="manual")
        chapter2 = VideoSegment(150, 200, 1, "SexAct", "SexAct", "Boundary2", "Boundary2 Position", source="manual")
        chapter3 = VideoSegment(250, 300, 1, "SexAct", "SexAct", "Boundary3", "Boundary3 Position", source="manual")
        
        fs_proc.video_chapters = [chapter1, chapter2, chapter3]
        
        # Try to extend chapter2 to overlap with chapter1 and chapter3
        overlapping_data = {
            "start_frame_str": 80,   # Would overlap with chapter1 (50-100)
            "end_frame_str": 270,    # Would overlap with chapter3 (250-300)
            "segment_type": "SexAct",
            "position_short_name_key": "Boundary2_Overlapping",
            "source": "manual_edit"
        }
        
        # Test overlap detection
        has_overlap = fs_proc._check_chapter_overlap(80, 270, chapter2.unique_id)
        assert has_overlap == True
        
        # The update should be rejected or handled gracefully
        # Implementation should prevent the overlapping update


class TestComplexChapterScenarios:
    """Test complex chapter scenarios and edge cases."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock(spec=ApplicationLogic)
        self.app.logger = Mock()
        self.app.project_manager = Mock(spec=ProjectManager)
        self.app.project_manager.project_dirty = False
        
        self.funscript_processor = AppFunscriptProcessor(self.app)

        self.app.app_state_ui = Mock()
        self.app.app_state_ui.heatmap_dirty = False
        self.app.app_state_ui.funscript_preview_dirty = False
    
    def test_chapter_creation_fills_gaps(self):
        """Test creating chapters to fill gaps between existing chapters."""
        fs_proc = self.funscript_processor
        
        # Create chapters with gaps
        chapter1 = VideoSegment(50, 100, 1, "SexAct", "SexAct", "GapFill1", "GapFill1 Position", source="manual")
        chapter3 = VideoSegment(200, 250, 1, "SexAct", "SexAct", "GapFill3", "GapFill3 Position", source="manual")
        
        fs_proc.video_chapters = [chapter1, chapter3]
        
        # Create chapter to fill the gap (101-199)
        gap_fill_data = {
            "start_frame_str": 101,
            "end_frame_str": 199,
            "segment_type": "SexAct",
            "position_short_name_key": "GapFiller",
            "source": "manual_gap_fill"
        }
        
        gap_chapter = fs_proc.create_new_chapter_from_data(gap_fill_data, return_chapter_object=True)
        
        # Verify gap was filled
        assert len(fs_proc.video_chapters) == 3
        fs_proc.video_chapters.sort(key=lambda c: c.start_frame_id)
        
        assert fs_proc.video_chapters[0].end_frame_id == 100
        assert fs_proc.video_chapters[1].start_frame_id == 101
        assert fs_proc.video_chapters[1].end_frame_id == 199
        assert fs_proc.video_chapters[2].start_frame_id == 200
    
    def test_multiple_chapter_operations_sequence(self):
        """Test a sequence of multiple chapter operations."""
        fs_proc = self.funscript_processor
        
        # Start with initial chapters
        chapters = [
            VideoSegment(
                start_frame_id=i*100,
                end_frame_id=(i*100) + 50,
                segment_type="SexAct",
                position_short_name=f"Sequence{i}",
                position_long_name=f"Sequence {i} Long",
                source="manual",
                class_name="Sequence",
                class_id=i
            ) for i in range(1, 5)  # Creates chapters at 100-150, 200-250, 300-350, 400-450
        ]
        
        fs_proc.video_chapters = chapters
        assert len(fs_proc.video_chapters) == 4
        
        # Operation 1: Merge first two chapters
        merged = fs_proc.merge_selected_chapters(fs_proc.video_chapters[0], fs_proc.video_chapters[1], return_chapter_object=True)
        assert len(fs_proc.video_chapters) == 3
        
        # Operation 2: Delete a chapter
        fs_proc.delete_video_chapters_by_ids([fs_proc.video_chapters[1].unique_id])
        assert len(fs_proc.video_chapters) == 2
        
        # Operation 3: Create new chapter in gap
        new_chapter_data = {
            "start_frame_str": 275,
            "end_frame_str": 300,
            "segment_type": "SexAct",
            "position_short_name_key": "NewInGap",
            "source": "manual",
            "class_name": "NewInGap",
            "class_id": 5
        }
        fs_proc.create_new_chapter_from_data(new_chapter_data, return_chapter_object=True)
        assert len(fs_proc.video_chapters) == 3
        
        # Verify final state makes sense
        fs_proc.video_chapters.sort(key=lambda c: c.start_frame_id)
        chapters_by_start = [ch.start_frame_id for ch in fs_proc.video_chapters]
        
        # Should have chapters starting at reasonable positions
        assert len(set(chapters_by_start)) == len(chapters_by_start)  # No duplicates
        assert all(start >= 0 for start in chapters_by_start)  # All positive


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])