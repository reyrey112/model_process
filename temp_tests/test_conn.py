import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect("postgresql://reyrey:reyrey@localhost:5432/reyrey")
    print("Connected!")
    await conn.close()

asyncio.run(main())