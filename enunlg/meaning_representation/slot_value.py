import box


class SlotValueMR(box.Box):
    def __init__(self, *args, **kwargs):
        super(SlotValueMR, self).__init__(*args, **kwargs)

    def __repr__(self):
        slot_value_pairs = ", ".join([f"{key}='{self[key]}'" for key in self.keys()])
        return f"{self.__class__.__name__}({slot_value_pairs})"

    def __str__(self):
        return self.__repr__()
