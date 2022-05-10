if __name__ == "__main__":

    with open("cleaned_raw_data.csv", "w") as f:
        with open("raw_data.csv", "r") as g:
            c = 0
            c1 = 0
            for l in g:
                # print("reading {}".format(l))

                if c == 0:
                    f.write(l)
                    # print(l)
                else:
                    l_list = l.split(",")
                    l_last = l_list[-1]
                    if l_last != " None\n":
                        s = ",".join(l_list)
                        assert s == l

                        # print(s)
                        # print("Writing {}".format(l))
                        if len(l_list) == 8:
                            c1 += 1
                            f.write(s)
                c += 1

    print("lenght file : ", c1)
