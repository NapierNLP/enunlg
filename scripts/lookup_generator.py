import enunlg.data_management.e2e_challenge as e2e
import enunlg as lug

if __name__ == "__main__":
    print("Loading E2E Challenge Data")
    splits =('trainset', )
    e2e_corpus = e2e.load_e2e(splits)
    print("----")
    print("Training a one-to-many lookup generator")
    generator = lug.OneToManyLookupGenerator()
    generator.train(e2e_corpus)
    print("----")
    print("Sampling predictions for 10 MRs.")
    for mr, orig_text in e2e_corpus:
        print(f"MR:   {mr}")
        print(f"Ref:  {orig_text}")
        print(f"Gen:  {generator.predict(mr)}")
        print("--")
