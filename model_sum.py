def main():
    import os
    from concurrent import futures

    path = os.path.dirname(os.path.abspath(__file__))
    python3 = "/usr/bin/python3"
    cmd = "nohup {python3} -u {py} >>{out} 2>&1 &"

    def exec(num: int):
        os.system(
            cmd.format(python3=python3,
                       py=os.path.join(path, "model_%s.py" % (num)),
                       out=os.path.join(path, "out", "model_%s.out" % (num))
                       )
        )

    with futures.ThreadPoolExecutor(max_workers=4) as exector:
        for i in range(1, 5):
            exector.submit(exec, i)


if __name__ == '__main__':
    main()

