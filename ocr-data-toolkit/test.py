from odt import ODT


if __name__ == "__main__":

    # print(Config.supported_languages)

    gen = ODT(
        num_samples=1,
        language="en",
    )
    gen.generate()
    