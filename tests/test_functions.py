import pytest
import math
from decimer_image_classifier import DecimerImageClassifier

decimer_classifier = DecimerImageClassifier()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_classifyChemicalStructure():
    img_path = "tests/caffeine.png"
    expected_result = True
    actual_result = decimer_classifier.is_chemical_structure(img_path=img_path)
    assert expected_result == actual_result


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_classifyNonChemicalStructure():
    img_path = "tests/chinese_character.jpg"
    expected_result = False
    actual_result = decimer_classifier.is_chemical_structure(img_path=img_path)
    assert expected_result == actual_result


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_scoreChemicalStructure():
    img_path = "tests/caffeine.png"
    expected_result = 0.000000016135415
    actual_result = decimer_classifier.get_classifier_score(img_path=img_path)
    # Set an appropriate tolerance value based on the required precision
    tolerance = 1e-9
    assert math.isclose(
        expected_result, actual_result, rel_tol=tolerance, abs_tol=tolerance
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_scoreNonChemicalStructure():
    img_path = "tests/chinese_character.jpg"
    expected_result = 1.0
    actual_result = decimer_classifier.get_classifier_score(img_path=img_path)
    assert expected_result == actual_result
