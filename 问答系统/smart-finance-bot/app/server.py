from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from finance_bot_ex import FinanceBotEx

finance_bot_ex = FinanceBotEx()
app = FastAPI(
    title='问答助手',
    version='0.1',
    description='问答助手API'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post("/queryex", response_model=dict)
async def query(query: dict):
    try:
        input_data = query.get('input')
        result = finance_bot_ex.handle_query(input_data)
        return {'output': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app=app, host='0.0.0.0', port=8082)
