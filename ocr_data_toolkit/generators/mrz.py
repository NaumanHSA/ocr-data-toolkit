import random
import string


class MRZGenerator:
    def __init__(
        self,
        num_samples: int,
        bag_of_words: List[str]
    ):
        self.num_samples = num_samples
        self.bag_of_words = bag_of_words
        self.bag_of_words.extend(self.generate_mrz_list(num_samples))
        
    def random_string(self, length):
        return ''.join(random.choices(string.ascii_uppercase, k=length))

    def random_name(self, length, num_words):
        surname = True
        name = ""
        count = 0
        while len(name) < length and count <= num_words:
            word__ = self.random_string(random.randint(3, 12))
            if surname:
                word__ += "<<"
                surname = False
            else:
                word__ += "<"
            name += word__
            if len(name) > length:
                name = name[:length]
            count += 1
        if len(name) < length:
            name = name + "<" * (length - len(name))
        return name

    def random_digits(self, length):
        return ''.join(random.choices(string.digits, k=length))

    def maybe_pad(self, content, total_length):
        if random.choice([True, False]):
            return content + '<' * (total_length - len(content))
        else:
            return content + self.random_string(total_length - len(content))

    def generate_mrz_td1(self):
        line1 = f"I{random.choice(['L', 'D'])}{''.join(random.choices(string.ascii_uppercase, k=3))}{self.random_digits(9)}"
        line1 = line1 + (self.random_digits(30 - len(line1)) if random.choice([True, False]) else "")
        line1 = self.maybe_pad(line1, 30)
        line2 = f"{self.random_digits(7)}{random.choice(['M', 'F'])}{self.random_digits(7)}{''.join(random.choices(string.ascii_uppercase, k=3))}"
        line2 = self.maybe_pad(line2, 30)
        line2 = line2[:-1] + "<" if random.choice([True, False]) else line2[:-1] + self.random_digits(1)
        line3 = self.random_name(30, random.randint(1, 3))
        return [line1, line2, line3]
        
    def generate_mrz_td3(self):
        line1 = f"P{random.choice(['M', 'D', 'O', '<', '<', '<'])}{''.join(random.choices(string.ascii_uppercase, k=3))}"
        line1 = line1 + self.random_name(44 - len(line1), random.randint(2, 4))
        # line1 = maybe_pad(line1, 44)

        line2 = f"{self.random_digits(10)}{''.join(random.choices(string.ascii_uppercase, k=3))}{self.random_digits(7)}{random.choice(['M', 'F'])}{self.random_digits(7)}"
        line2 = line2 + (self.random_digits(44 - len(line2)) if random.choice([True, False]) else "")
        line2 = self.maybe_pad(line2, 44)
        return [line1, line2]

    def generate_mrz_list(self, n):
        mrz_list = []
        for _ in range(n):
            mrz_list.extend(random.choice([self.generate_mrz_td1, self.generate_mrz_td3])())
        return mrz_list

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass