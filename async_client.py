import json
import joblib
import asyncio
import aiohttp

FOLDER_WITH_MODELS = '../apartmentML/models/'

# Dict with information about sklearn preprocessing objects
transformer = joblib.load(FOLDER_WITH_MODELS+'transformers_info')

# Data to send
d = json.dumps({'Cost': 96000, 'DistanceToCenter': 3.4, 'Area': 126, 'Floor': 2, 'BuildingType': 'New building',
                'LivingArea': 96, 'KitchenArea': 22, 'Condition': 'luxury', 'WallsMaterial': 'brick',
                'Balconies': 0, 'CeilingHeight': 2.75, 'Floors': 5})


async def send_data(data):
    async with aiohttp.ClientSession() as session:
        result = await session.get('http://localhost:3030/predictRooms', json=data)
        return await result.text()


loop = asyncio.get_event_loop()
predicted_value = loop.run_until_complete(send_data(d))

print(predicted_value)