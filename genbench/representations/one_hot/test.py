import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from genbench.data.schema import TabularSchema
from genbench.representations.one_hot.one_hot import OneHotRepresentation


class BaseRepresentationTest(unittest.TestCase):
    """Абстрактный тест для всех классов представлений."""

    representation_class = OneHotRepresentation
    representation_kwargs = {}  # optional params
    requires_fit_expected = True
    is_invertible_expected = True

    @classmethod
    def setUpClass(cls):
        """Проверка, что представление определено."""
        if cls.representation_class is None:
            raise unittest.SkipTest(
                "BaseRepresentationTest не должен запускаться напрямую")

    def setUp(self):
        """Синтетические данные для обучения и тестирования."""
        self.train_df = pd.DataFrame({
            'color': ['red', 'blue', 'green', 'red'],
            'size': ['small', 'medium', 'large', 'small'],
            'price': [10.0, 15.0, 20.0, 12.0]
        })
        # Явно указываем типы колонок
        self.schema = TabularSchema.infer_from_dataframe(
            self.train_df,
            categorical_cols=['color', 'size'],
            continuous_cols=['price']
        )

        self.test_df = pd.DataFrame({
            'color': ['purple', 'red'],
            'size': ['tiny', 'small'],
            'price': [11.0, 13.0]
        })

        self.rep = self.representation_class(**self.representation_kwargs)

    # ----- Тесты интерфейса -----
    def test_methods_exist(self):
        """Проверка наличия обязательных методов."""
        methods = ['fit', 'transform', 'inverse_transform', 'requires_fit',
                   'is_invertible', 'get_state', 'from_state']
        for method in methods:
            self.assertTrue(hasattr(self.rep, method),
                            f"Отсутствует метод {method}")

    def test_requires_fit(self):
        """Проверка значения requires_fit."""
        self.assertEqual(self.rep.requires_fit(), self.requires_fit_expected)

    def test_is_invertible(self):
        """Проверка значения is_invertible."""
        self.assertEqual(self.rep.is_invertible(), self.is_invertible_expected)

    def test_fit_sets_fitted_flag(self):
        """После fit должен быть установлен fitted_ = True."""
        self.rep.fit(self.train_df, self.schema)
        self.assertTrue(self.rep.fitted_)

    def test_transform_requires_fit(self):
        """transform без fit должен кидать ошибку."""
        with self.assertRaises(RuntimeError):
            self.rep.transform(self.train_df)

    def test_inverse_transform_requires_fit(self):
        """inverse_transform без fit должен кидать ошибку."""
        with self.assertRaises(RuntimeError):
            self.rep.inverse_transform(self.train_df)

    def test_transform_returns_dataframe(self):
        """transform должен возвращать pandas DataFrame."""
        self.rep.fit(self.train_df, self.schema)
        result = self.rep.transform(self.train_df)
        self.assertIsInstance(result, pd.DataFrame)

    # ----- Тесты сериализации -----
    def test_get_state_from_state_roundtrip(self):
        """Проверка, что get_state -> from_state восстанавливает объект."""
        self.rep.fit(self.train_df, self.schema)
        state = self.rep.get_state()
        new_rep = self.representation_class.from_state(state)
        # Сравниваем состояние
        self.assertEqual(state.params, new_rep.get_state().params)

    # ----- Тесты обратного преобразования (если invertible) -----
    def test_inverse_transform_roundtrip_on_train(self):
        """Если invertible, проверяем восстановление на train."""
        if not self.rep.is_invertible():
            self.skipTest("Представление необратимо")
        self.rep.fit(self.train_df, self.schema)
        transformed = self.rep.transform(self.train_df)
        recovered = self.rep.inverse_transform(transformed)
        # Сравниваем оригинальные категориальные колонки
        for col in self.schema.categorical_cols:
            pd.testing.assert_series_equal(
                recovered[col], self.train_df[col], check_names=False
            )

    def test_unknown_category_handling(self):
        assert 0 == 1
        """Базовая проверка обработки неизвестных категорий."""
        self.rep.fit(self.train_df, self.schema)
        transformed = self.rep.transform(self.test_df)
        # Должен быть валидный DataFrame
        self.assertIsInstance(transformed, pd.DataFrame)
        # Обратное преобразование не должно падать
        if self.rep.is_invertible():
            recovered = self.rep.inverse_transform(transformed)
            self.assertEqual(len(recovered), len(self.test_df))


if __name__ == '__main__':
    unittest.main()
