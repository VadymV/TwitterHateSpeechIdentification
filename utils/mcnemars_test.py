from statsmodels.stats.contingency_tables import mcnemar


class McNemarsTest:
    def __init__(self, data):
        self.data = data
        self.c_table = self.create_contingency_table()

    def create_contingency_table(self):
        true_true = self.data.loc[(self.data['correct_preds_cls1'] == 1) &
                                  (self.data['correct_preds_cls2'] == 1)].shape[0]
        true_false = self.data.loc[(self.data['correct_preds_cls1'] == 1) &
                                   (self.data['correct_preds_cls2'] == 0)].shape[0]
        false_true = self.data.loc[(self.data['correct_preds_cls1'] == 0) &
                                   (self.data['correct_preds_cls2'] == 1)].shape[0]
        false_false = self.data.loc[(self.data['correct_preds_cls1'] == 0) &
                                    (self.data['correct_preds_cls2'] == 0)].shape[0]

        table = [[true_true, true_false],
                 [false_true, false_false]]

        print(table)
        return table

    def calculate_value(self):
        result = mcnemar(self.c_table, exact=False, correction=True)
        print('statistic is %.2f, p-value is %.2f' % (result.statistic, result.pvalue))
        threshold = 0.05
        if result.pvalue > threshold:
            print('H0 is not rejected')
        else:
            print('Reject H0')


