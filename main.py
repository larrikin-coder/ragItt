from fastapi import FastAPI


app = FastAPI(name="ragItt")


@app.get("/health")
def health_check():
    return {"Status":"ok"}



def run():
    pass


if __name__ == "__main__":
    run() 