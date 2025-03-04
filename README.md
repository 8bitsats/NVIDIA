# NVIDIA

NVIDIA NIM (NVIDIA Inference Microservices) is a powerful set of tools designed to accelerate the deployment of generative AI models across various platforms, including clouds, data centers, and workstations. It provides optimized, containerized microservices that simplify the integration of AI into applications, making it easier for developers and enterprises to build and scale AI-powered solutions. Below, I’ll outline some key applications of NVIDIA NIM based on its capabilities and real-world use cases.

### Key Applications of NVIDIA NIM

1. **Customer Service Automation**
   - **Use Case**: NIM powers digital human avatars and conversational AI agents for customer support.
   - **How It Works**: Using NIM microservices like NVIDIA ACE (Avatar Creation Engine) and Tokkio, businesses can deploy lifelike, interactive avatars that handle customer inquiries in real time. These avatars leverage natural language processing (NLP) and real-time lip-syncing for natural, human-like interactions.
   - **Example**: A retail company could use a NIM-powered digital human to assist customers with product queries, reducing wait times and improving satisfaction.

2. **Retrieval-Augmented Generation (RAG)**
   - **Use Case**: Enhancing chatbots and AI assistants with accurate, context-aware responses.
   - **How It Works**: NIM integrates with NVIDIA NeMo Retriever microservices to enable RAG workflows, allowing AI agents to fetch and incorporate proprietary or up-to-date data into their responses. This is ideal for applications requiring domain-specific knowledge.
   - **Example**: An enterprise could deploy a RAG-based chatbot to answer employee questions about internal policies by pulling data from company PDFs and databases.

3. **Healthcare and Drug Discovery**
   - **Use Case**: Accelerating drug discovery and medical research.
   - **How It Works**: NIM microservices like BioNeMo support generative AI models for biology and chemistry, enabling researchers to generate molecular structures or predict protein interactions. These models run efficiently on NVIDIA GPUs, speeding up virtual screening processes.
   - **Example**: A pharmaceutical company might use NIM to screen millions of compounds virtually, identifying potential drug candidates faster than traditional methods.

4. **Manufacturing and Industrial Automation**
   - **Use Case**: Optimizing factory operations and automation workflows.
   - **How It Works**: NIM enables AI-driven solutions for smart manufacturing, such as predictive maintenance and quality control, by deploying models optimized for NVIDIA hardware. It integrates with platforms like NVIDIA Metropolis for real-time analytics.
   - **Example**: Companies like Foxconn use NIM to enhance AI factories, improving efficiency in smart manufacturing and electric vehicle production.

5. **Gaming and Entertainment**
   - **Use Case**: Creating interactive characters and virtual experiences.
   - **How It Works**: NIM supports multimodal models (e.g., Audio2Face-3D) that animate avatars based on audio inputs, enabling lifelike NPCs (non-player characters) or virtual assistants in games and media.
   - **Example**: Game developers could use NIM to bring dynamic, responsive characters to life, enhancing player immersion.

6. **Multimodal Data Extraction**
   - **Use Case**: Extracting insights from unstructured data like PDFs, images, or videos.
   - **How It Works**: NIM’s multimodal capabilities (e.g., Cosmos Nemotron vision models) allow AI agents to process and interpret diverse data types, making it valuable for enterprise knowledge management.
   - **Example**: A legal firm could use NIM to extract key information from contracts in PDF format, streamlining document review processes.

7. **Agentic AI Workflows**
   - **Use Case**: Building autonomous AI agents for complex tasks.
   - **How It Works**: NIM Agent Blueprints provide prebuilt, customizable workflows for tasks like customer service, virtual screening, or data analysis. These blueprints leverage models like Llama Nemotron for reasoning and multistep problem-solving.
   - **Example**: A logistics company might deploy an AI agent to optimize delivery routes using real-time data and predictive analytics.

8. **Creative Content Generation**
   - **Use Case**: Enhancing workflows in design, media, and marketing.
   - **How It Works**: NIM integrates with tools like NVIDIA Picasso and Edify models to generate high-quality images, 3D models, or videos, accelerating creative production.
   - **Example**: A marketing agency could use NIM to create tailored visual content for campaigns, leveraging licensed libraries for compliance.

9. **Weather and Environmental Modeling**
   - **Use Case**: Improving forecasting and climate analysis.
   - **How It Works**: NIM supports models like CorrDiff for weather data downscaling, enhancing prediction accuracy with generative AI techniques.
   - **Example**: A meteorological agency could use NIM to refine weather models, providing more precise local forecasts.

10. **Enterprise Copilots and Productivity Tools**
    - **Use Case**: Assisting employees with daily tasks.
    - **How It Works**: NIM enables the deployment of AI copilots (e.g., using Llama 3.1 or Mixtral models) that integrate with existing workflows via standard APIs, boosting productivity.
    - **Example**: A software company might deploy a NIM-powered coding assistant to help developers write and debug code faster.

### Why NVIDIA NIM Stands Out
- **Scalability**: NIM can run on any NVIDIA GPU-powered infrastructure, from edge devices to large data centers, making it highly adaptable.
- **Ease of Use**: Prebuilt containers and industry-standard APIs (e.g., OpenAI-compatible) reduce deployment time from weeks to minutes.
- **Performance**: Optimized inference engines (TensorRT, TensorRT-LLM) ensure low latency and high throughput, critical for real-time applications.
- **Security**: Self-hosted options keep sensitive data within enterprise boundaries, a key requirement for industries like healthcare and finance.

### Real-World Impact
- **Foxconn**: Uses NIM for domain-specific LLMs in smart manufacturing, integrating AI into production lines.
- **ServiceNow**: Leverages NIM for generative AI in customer service and workflow automation.
- **Amdocs**: Deploys NIM to run billing LLMs, cutting costs and latency while improving accuracy.

NVIDIA NIM’s versatility makes it a game-changer for industries looking to harness generative AI quickly and efficiently. Whether you’re building a chatbot, analyzing enterprise data, or advancing scientific research, NIM provides the tools to turn ideas into production-ready solutions. Let me know if you’d like a deeper dive into any specific application!
### **Step-by-Step Guide: Integrating an Exo Labs Cluster with the reComputer J3011 System for Solana Blockchain Trading**

In this guide, we’ll extend the **reComputer J3011 system** to integrate an **Exo Labs cluster**. Exo Labs provides a powerful platform for distributed computing, which can enhance the scalability and performance of your vision-based trading system. The Exo Labs cluster will handle heavy computational tasks, while the reComputer J3011 devices focus on real-time trading and vision processing.

---

### **Step 1: Set Up the Exo Labs Cluster**
1. **Sign Up for Exo Labs**:
   - Visit the [Exo Labs website](https://www.exo.io/) and create an account.
   - Follow the onboarding process to set up your cluster.

2. **Deploy the Exo Labs Cluster**:
   - Use the Exo Labs dashboard to deploy a new cluster.
   - Choose the appropriate configuration (e.g., number of nodes, GPU support) based on your computational needs.

3. **Install Exo Labs CLI**:
   - On your Mac, install the Exo Labs CLI:
     ```bash
     curl -sSL https://get.exo.io/cli | sh
     ```
   - Authenticate the CLI with your Exo Labs account:
     ```bash
     exo auth login
     ```

4. **Connect to the Cluster**:
   - Use the Exo Labs CLI to connect to your cluster:
     ```bash
     exo cluster connect <cluster-name>
     ```

---

### **Step 2: Configure the Exo Labs Cluster for Vision Processing**
1. **Install Required Libraries**:
   - On the Exo Labs cluster, install Python, OpenCV, and PyTorch:
     ```bash
     sudo apt update
     sudo apt install python3-pip python3-opencv -y
     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
     ```

2. **Deploy Vision Models**:
   - Upload your pre-trained vision models (e.g., YOLOv8) to the Exo Labs cluster.
   - Use the Exo Labs CLI to deploy the models:
     ```bash
     exo deploy model --name chart-analysis --path /path/to/model
     ```

3. **Set Up a Vision API**:
   - Use Flask or FastAPI to create a REST API for vision processing on the Exo Labs cluster:
     ```python
     from flask import Flask, request, jsonify
     import cv2
     import torch

     app = Flask(__name__)
     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

     @app.route('/analyze', methods=['POST'])
     def analyze():
         image = request.files['image'].read()
         results = model(image)
         return jsonify(results.pandas().xyxy[0].to_dict())

     if __name__ == '__main__':
         app.run(host='0.0.0.0', port=5000)
     ```
   - Deploy the API using Exo Labs:
     ```bash
     exo deploy api --name vision-api --path /path/to/api
     ```

---

### **Step 3: Integrate the Exo Labs Cluster with the reComputer J3011 System**
1. **Modify Vision Script on reComputer**:
   - Update the vision script on the reComputer to send images to the Exo Labs cluster for processing:
     ```python
     import requests

     def analyze_chart(image_path):
         url = "http://<exo-cluster-ip>:5000/analyze"
         with open(image_path, 'rb') as image_file:
             response = requests.post(url, files={'image': image_file})
         return response.json()
     ```

2. **Test the Integration**:
   - Run the updated vision script on the reComputer and verify that it communicates with the Exo Labs cluster.

---

### **Step 4: Enhance the Swarm with Exo Labs**
1. **Distribute Workloads**:
   - Use the Exo Labs cluster to handle computationally intensive tasks (e.g., training vision models, backtesting trading strategies).
   - Offload these tasks from the reComputer devices to improve real-time performance.

2. **Set Up Swarm Coordination**:
   - Use MQTT or ZeroMQ to coordinate tasks between the reComputer devices and the Exo Labs cluster.
   - Deploy a coordination script on your Mac to manage the swarm:
     ```python
     import paho.mqtt.client as mqtt

     def on_message(client, userdata, message):
         task = message.payload.decode()
         if task == "process_image":
             # Send image to Exo Labs cluster
             pass
         elif task == "train_model":
             # Train model on Exo Labs cluster
             pass

     client = mqtt.Client()
     client.connect("localhost", 1883, 60)
     client.subscribe("swarm/commands")
     client.on_message = on_message
     client.loop_forever()
     ```

---

### **Step 5: Monitor and Optimize**
1. **Monitor Cluster Performance**:
   - Use the Exo Labs dashboard to monitor the performance of your cluster.
   - Adjust the number of nodes or resources as needed.

2. **Optimize Vision Models**:
   - Continuously train and fine-tune your vision models on the Exo Labs cluster.
   - Deploy updated models to the reComputer devices as needed.

---

### **Step 6: Secure the System**
1. **Enable Firewall on Exo Labs Cluster**:
   - Use the Exo Labs dashboard to configure firewall rules and restrict access to the cluster.

2. **Use HTTPS for API Communication**:
   - Configure your vision API to use HTTPS for secure communication.

---

### **Step 7: Deploy and Scale**
1. **Deploy the System**:
   - Ensure all components (reComputer devices, Exo Labs cluster, Mac hub) are connected and functioning correctly.
2. **Scale the System**:
   - Add more reComputer devices or Exo Labs nodes as needed to handle increased workloads.

---
Below is a step-by-step guide to set up the Conversational AI framework using the Pipecat AI framework as outlined in your request. This guide is tailored to the specific notebook example you provided, which uses NVIDIA NIM, Riva STT/TTS, and the Daily WebRTC transport layer with the `meta/llama-3.3-70b-instruct` LLM model. I'll break it down into clear, actionable steps while ensuring it’s beginner-friendly yet detailed enough for practical implementation.

---

### Step-by-Step Guide to Set Up the Conversational AI Framework with Pipecat AI

#### Step 1: UnderstandPrerequisites
Before you begin, ensure you have the necessary accounts and tools:
1. **NVIDIA API Key**: Required for NVIDIA NIM services (LLM, STT, TTS).
   - Sign up at the [NVIDIA API Catalog](https://catalog.ngc.nvidia.com/).
   - Select a model (e.g., `llama-3.3-70b-instruct`), click "Get API Key," and log in.
2. **Daily API Key**: Needed for WebRTC transport.
   - Sign up at [Daily](https://www.daily.co/), verify your email, choose a subdomain, and get your API key from the "Developers" section in the dashboard.
3. **Python Environment**: Ensure you have Python 3.8+ installed.
4. **System Requirements**: A machine with internet access and sufficient memory (8GB+ RAM recommended).

---

#### Step 2: Set Up Your Environment
1. **Create a Virtual Environment** (optional but recommended):
   - Open a terminal and navigate to your project directory.
   - Run:
     ```bash
     python3 -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
2. **Export API Keys as Environment Variables**:
   - In your terminal, set the NVIDIA API Key:
     ```bash
     export NVIDIA_API_KEY="nvapi-your-key-here"
     ```
   - Set the Daily API Key:
     ```bash
     export DAILY_API_KEY="your-64-character-daily-key-here"
     ```
   - Alternatively, you can hardcode these in your script (not recommended for security reasons).

3. **Install Dependencies**:
   - Install the Pipecat framework with required services:
     ```bash
     pip install "pipecat-ai[daily,openai,riva,silero]"
     ```
   - Install the NOAA SDK for the weather tool example:
     ```bash
     pip install noaa_sdk
     ```

---

#### Step 3: Create the Python Script
Create a file (e.g., `agent.py`) and add the following code. This script implements the framework as described.

```python
import aiohttp
import asyncio
import os
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.nim import NimLLMService
from pipecat.services.riva import FastPitchTTSService, ParakeetSTTService
from pipecat.transports.services.daily import DailyParams, DailyTransport, DailyRESTHelper, DailyRoomParams
from openai.types.chat import ChatCompletionToolParam
from noaa_sdk import NOAA

# Step 1: Create a Daily Room
async def create_room():
    async with aiohttp.ClientSession() as session:
        daily_rest_helper = DailyRESTHelper(
            daily_api_key=os.getenv("DAILY_API_KEY"),
            daily_api_url="https://api.daily.co/v1",
            aiohttp_session=session,
        )
        room_config = await daily_rest_helper.create_room(
            DailyRoomParams(properties={"enable_prejoin_ui": False})
        )
        return room_config.url

# Step 2: Define Weather Tool Functions
async def start_fetch_weather(function_name, llm, context):
    print(f"Starting fetch_weather_from_api with function_name: {function_name}")

async def get_noaa_simple_weather(latitude: float, longitude: float):
    n = NOAA()
    description = False
    fahrenheit_temp = 0
    try:
        observations = n.get_observations_by_lat_lon(latitude, longitude, num_of_stations=1)
        for observation in observations:
            description = observation["textDescription"]
            celcius_temp = observation["temperature"]["value"]
            if description:
                break
        fahrenheit_temp = (celcius_temp * 9 / 5) + 32
        if fahrenheit_temp and not description:
            description = fahrenheit_temp
    except Exception as e:
        print(f"Error getting NOAA weather: {e}")
    return description, fahrenheit_temp

async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    location = args["location"]
    latitude = float(args["latitude"])
    longitude = float(args["longitude"])
    description, fahrenheit_temp = await get_noaa_simple_weather(latitude, longitude)
    if not description:
        await result_callback(f"I'm sorry, I can't get the weather for {location} right now. Can you ask again please?")
    else:
        await result_callback(f"The weather in {location} is currently {round(fahrenheit_temp)} degrees and {description}.")

# Step 3: Main Function to Set Up and Run the Agent
async def main():
    # Get Daily Room URL
    DAILY_ROOM_URL = await create_room()
    print(f"Navigate to: {DAILY_ROOM_URL}")

    # Configure Daily Transport
    transport = DailyTransport(
        DAILY_ROOM_URL,
        None,
        "Lydia",
        DailyParams(
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    # Initialize Services
    stt = ParakeetSTTService(api_key=os.getenv("NVIDIA_API_KEY"))
    llm = NimLLMService(api_key=os.getenv("NVIDIA_API_KEY"), model="meta/llama-3.3-70b-instruct")
    tts = FastPitchTTSService(api_key=os.getenv("NVIDIA_API_KEY"))

    # Define LLM Prompt
    messages = [
        {
            "role": "system",
            "content": """
Hello, I'm Lydia. I'm looking forward to talking about in vidia's recent work in agentic AI. I can also demonstrate tool use by responding to questions about the current weather anywhere in the United States. Who am I speaking with?
You are Lydia; a conversational voice agent who discusses Nvidia's work in agentic AI and a sales assistant who listens to the user and answers their questions. If you are asked how you were built, say you were built with the pipe cat framework and the in vidia NIM platform.
INSTRUCTIONS: Answer questions about in vidia's work in agentic AI, discuss its impact on industries, and provide weather info for the United States only.
""",
        }
    ]

    # Define Weather Tool
    tools = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location for the weather request."},
                        "latitude": {"type": "string", "description": "Latitude as a string, e.g., '42.3601'."},
                        "longitude": {"type": "string", "description": "Longitude as a string, e.g., '-71.0589'."},
                    },
                    "required": ["location", "latitude", "longitude"],
                },
            },
        )
    ]
    llm.register_function(None, fetch_weather_from_api, start_callback=start_fetch_weather)

    # Set Up Context Aggregator
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # Create Pipeline
    pipeline = Pipeline([
        transport.input(),          # User audio input
        stt,                        # Speech-to-Text
        context_aggregator.user(),  # User context
        llm,                        # LLM processing
        tts,                        # Text-to-Speech
        transport.output(),         # Audio output
        context_aggregator.assistant(),  # Assistant context
    ])

    # Create Pipeline Task
    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    # Set Event Handlers
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        await task.queue_frame(EndFrame())

    # Run the Pipeline
    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
```

---

#### Step 4: Run the Agent
1. **Save the Script**: Save the file as `agent.py`.
2. **Run the Script**:
   - In your terminal (with the virtual environment activated), run:
     ```bash
     python agent.py
     ```
   - The script will output a Daily room URL (e.g., `https://yourdomain.daily.co/room-name`).
3. **Join the Room**:
   - Open a web browser and navigate to the URL printed in the terminal.
   - Allow microphone access when prompted.
   - The agent (Lydia) will greet you and begin the conversation.

---

#### Step 5: Test the Agent
- **Suggested Interactions**:
  - Ask: "What is NVIDIA doing in agentic AI?"
  - Ask: "What's the weather in New York?"
  - After a few exchanges, ask: "What was the first thing I said?"
- **End the Call**: Close the browser tab or leave the Daily room to stop the agent.

---

#### Step 6: Troubleshooting
- **No Audio**: Ensure your microphone is enabled and the Daily API key is correct.
- **Errors with NVIDIA Services**: Verify your NVIDIA API key and internet connection.
- **First Run Delay**: The initial 10-15 seconds delay is normal as the VAD model loads.

---

#### Step 7: Customize (Optional)
- **Change LLM Model**: Modify `model="meta/llama-3.3-70b-instruct"` in the `NimLLMService` instantiation to another supported model.
- **Adjust Prompt**: Edit the `messages` content to change Lydia’s behavior or knowledge base.
- **Local NIM Microservices**: Replace the API endpoints with `base_url="http://localhost:8000/v1"` if running NIM locally.

---

### Key Points
- Research suggests NeMo and Riva are key NVIDIA tools for Speech AI, focusing on training and deploying ASR and TTS models.
- It seems likely that NeMo trains and customizes models, while Riva optimizes and deploys them for real-time use.
- The evidence leans toward their integration enabling efficient, scalable speech applications like virtual assistants and transcription services.

---

### Overview of NeMo and Riva in Speech AI

**What Are NeMo and Riva?**  
NeMo, or NVIDIA's Neural Modules, is a framework for building and training AI models, including those for Speech AI like Automatic Speech Recognition (ASR) and Text-to-Speech (TTS). Riva, on the other hand, is a GPU-accelerated SDK designed to deploy these models in production, ensuring real-time performance for applications like virtual assistants and call centers.

**Roles in Speech AI**  
- **NeMo**: Used for training and customizing ASR and TTS models, leveraging GPU clusters for efficiency. It supports fine-tuning models on custom datasets, making them domain-specific.  
- **Riva**: Takes optimized NeMo models and deploys them as microservices, using tools like TensorRT for low-latency, high-throughput inference. It supports multiple languages and scales to handle thousands of concurrent users.

**How They Work Together**  
NeMo handles the training phase, where developers create or fine-tune speech models. These models are then optimized for inference and deployed via Riva, ensuring they perform efficiently in real-world applications. This partnership allows for seamless transitions from research to production, supporting use cases like customer service automation and voice-activated devices.

**Unexpected Detail: Multilingual and Domain-Specific Capabilities**  
An interesting aspect is how both tools now support multiple languages and domain-specific customization, enabling applications in diverse global markets, such as healthcare or finance, with tailored speech recognition and synthesis.

---

---

### Survey Note: Comprehensive Analysis of NeMo and Riva in Speech AI

This note provides a detailed examination of NVIDIA's NeMo framework and Riva SDK, focusing on their roles in Speech AI, particularly Automatic Speech Recognition (ASR) and Text-to-Speech (TTS). It expands on their integration, recent advancements, and practical applications, ensuring a thorough understanding for researchers, developers, and enterprises interested in deploying speech technologies.

#### Introduction to NeMo and Riva

**NVIDIA NeMo Framework**  
NeMo, or NVIDIA's Neural Modules, is a scalable and cloud-native generative AI framework built for researchers and developers. It supports the development of Large Language Models (LLMs), Multimodal Models, and Speech AI, with a strong emphasis on ASR and TTS. According to the official documentation ([NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/index.html)), NeMo leverages PyTorch and PyTorch Lightning for efficient multi-GPU and multi-node training, making it ideal for handling large-scale data and computing resources. It enables users to create, customize, and deploy new generative AI models by leveraging existing code and pretrained model checkpoints, available on platforms like Hugging Face Hub and NVIDIA NGC.

**NVIDIA Riva SDK**  
Riva is a GPU-accelerated SDK designed for building and deploying Speech AI applications with real-time performance. As detailed in the Riva user guide ([Riva — NVIDIA Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html)), it offers pretrained speech models that can be fine-tuned with NeMo on custom datasets, accelerating development by up to 10x. Riva uses NVIDIA TensorRT for optimizations and NVIDIA Triton Inference Server for serving, ensuring low-latency streaming and high-throughput offline use cases. It is fully containerized, scalable to hundreds of thousands of concurrent users, and deployable on-premises, in the cloud, or at the edge.

#### Roles in Speech AI

**NeMo's Role in Speech AI**  
NeMo is pivotal for training and customizing ASR and TTS models. It provides end-to-end support for developing speech models, utilizing native PyTorch and PyTorch Lightning for seamless integration and ease of use, as noted in the NeMo user guide ([Introduction — NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/intro.html)). It supports synthetic data generation and data augmentation techniques to enhance model robustness, particularly for speech processing. NeMo's focus areas include speech recognition, where it trains models to transcribe audio into text, and TTS, where it generates human-like speech from text. Recent updates, such as support for Llama 3.1 and distributed training on Amazon EKS, further enhance its capabilities ([GitHub - NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)).

**Riva's Role in Speech AI**  
Riva complements NeMo by focusing on deployment and inference. It provides microservices for ASR, TTS, and neural machine translation (NMT), optimized for real-time performance. According to the Riva developer page ([Riva Speech AI SDK - Get Started | NVIDIA Developer](https://developer.nvidia.com/riva)), Riva includes features like low-code AI custom voice creation, offline speaker diarization, and speech hints API for improved accuracy on specific content like addresses. It supports multilingual applications, with models for languages like English, Spanish, German, Russian, and Mandarin, as highlighted in a technical blog ([Build Speech AI in Multiple Languages and Train Large Language Models with the Latest from Riva and NeMo Framework | NVIDIA Technical Blog](https://developer.nvidia.com/blog/build-speech-ai-in-multiple-languages-and-train-large-language-models-with-the-latest-from-riva-and-nemo-megatron/)).

#### Integration and Workflow

The integration of NeMo and Riva creates a seamless workflow from research to production in Speech AI. Developers use NeMo to train or fine-tune ASR and TTS models, leveraging its extensive recipes and utilities, such as those in the NeMo Framework Launcher, although currently limited to NeMo 1.0 for LLMs and soon to support ASR/TTS training ([GitHub - NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)). Once trained, models are optimized for inference using TensorRT, a process detailed in Riva's documentation ([Overview — NVIDIA Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/overview.html)). Riva then deploys these models as microservices, accessible via gRPC-based APIs, ensuring low-latency streaming for real-time applications.

A practical example of this workflow is provided in a tutorial for TTS fine-tuning using NeMo, which assumes familiarity with NeMo and prepares models for export to .riva format for deployment ([Text to Speech Finetuning using NeMo — NVIDIA Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/tts-finetune-nemo.html)). This integration is crucial for enterprises looking to deploy speech AI at scale, as seen in use cases like customer care centers and virtual assistants, with industry leaders like Snap and T-Mobile utilizing Riva ([Build Speech AI in Multiple Languages and Train Large Language Models with the Latest from Riva and NeMo Framework | NVIDIA Technical Blog](https://developer.nvidia.com/blog/build-speech-ai-in-multiple-languages-and-train-large-language-models-with-the-latest-from-riva-and-nemo-megatron/)).

#### Recent Advancements and Features

Recent advancements in NeMo and Riva have significantly enhanced Speech AI capabilities. NeMo now supports training and customizing models like Llama 3.1, with distributed training workloads on Amazon EKS, as announced in July 2024 ([GitHub - NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)). It also includes Cosmos tokenizers for efficient mapping of visual data, although primarily focused on multimodal applications, its speech AI components continue to evolve ([GitHub - NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)).

Riva, on the other hand, has seen updates like Riva 2.0, announced at GTC 2022, with improved ASR and TTS models supporting seven languages and domain-specific customization using NeMo or TAO Toolkit ([Nvidia Upgrades Speech AI to Pursue Enterprise Ambitions - Voicebot.ai](https://voicebot.ai/2022/03/22/nvidia-upgrades-speech-ai-to-pursue-enterprise-ambitions/)). Recent blogs highlight Riva's role in integrating with LLMs and RAG for multilingual assistants, enhancing conversational AI ([Unlocking the Power of Speech AI: A Step-by-Step Guide to Integrating NVIDIA RIVA NIMs with LLM/RAG Applications | by Eda Johnson | Medium](https://edemiraydin.medium.com/unlocking-the-power-of-speech-ai-a-step-by-step-guide-to-integrating-nvidia-riva-nims-with-llm-rag-95bd92fe06a7)).

#### Use Cases and Practical Applications

The combined use of NeMo and Riva has led to numerous practical applications in Speech AI:

- **Customer Service Automation:** Deploying AI-powered virtual assistants that handle customer queries in real time, with natural responses despite background noise, as seen in deployments by T-Mobile and RingCentral ([Build Speech AI in Multiple Languages and Train Large Language Models with the Latest from Riva and NeMo Framework | NVIDIA Technical Blog](https://developer.nvidia.com/blog/build-speech-ai-in-multiple-languages-and-train-large-language-models-with-the-latest-from-riva-and-nemo-megatron/)).
- **Transcription Services:** Converting spoken language to text for meeting minutes, video captions, and accessibility features, leveraging Riva's high-accuracy ASR models.
- **Voice-Activated Devices:** Powering smart home devices and wearables with natural language understanding and response, utilizing Riva's TTS for human-like speech synthesis.
- **Multilingual Support:** Enabling global enterprises to serve diverse linguistic markets, with examples like AI robots in hospitals and retail stores using Riva for speech and translation ([Riva | Speech and Translation AI | NVIDIA](https://www.nvidia.com/en-us/ai-data-science/products/riva/)).

#### Comparative Analysis

To illustrate the roles and capabilities, here's a table comparing NeMo and Riva in the context of Speech AI:

| **Aspect**               | **NeMo**                                      | **Riva**                                      |
|--------------------------|-----------------------------------------------|-----------------------------------------------|
| **Primary Function**      | Training and customizing ASR/TTS models       | Deploying and optimizing models for production |
| **Key Technology**        | PyTorch, PyTorch Lightning, Megatron-LM       | TensorRT, Triton Inference Server             |
| **Use Case Focus**        | Research, model development, fine-tuning      | Real-time inference, large-scale deployment   |
| **Scalability**           | Multi-GPU, multi-node training                | Scales to thousands of concurrent streams     |
| **Languages Supported**   | Training for multiple languages               | Deployment in English, Spanish, German, etc.  |
| **Deployment Environment**| On-premises, cloud, Kubernetes, SLURM         | Cloud, on-premises, edge, embedded devices    |

This table highlights their complementary nature, with NeMo focusing on the development phase and Riva on the deployment phase, ensuring a complete pipeline for Speech AI applications.

#### Conclusion

NeMo and Riva together form a robust ecosystem for Speech AI, from training and customizing models with NeMo to deploying them efficiently with Riva. Their integration supports a wide range of applications, from customer service bots to voice-activated devices, with recent advancements enhancing multilingual and domain-specific capabilities. This synergy is particularly valuable for enterprises seeking to leverage speech technologies at scale, as evidenced by industry use cases and continuous updates in 2024 and early 2025.

---

### Key Citations
- [NVIDIA NeMo Framework User Guide Overview](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)
- [Riva — NVIDIA Riva User Guide](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html)
- [Introduction — NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/intro.html)
- [GitHub NVIDIA NeMo Repository](https://github.com/NVIDIA/NeMo)
- [Riva Speech AI SDK Get Started NVIDIA Developer](https://developer.nvidia.com/riva)
- [Build Speech AI in Multiple Languages NVIDIA Technical Blog](https://developer.nvidia.com/blog/build-speech-ai-in-multiple-languages-and-train-large-language-models-with-the-latest-from-riva-and-nemo-megatron/)
- [Text to Speech Finetuning using NeMo NVIDIA Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/tts-finetune-nemo.html)
- [Unlocking the Power of Speech AI Medium](https://edemiraydin.medium.com/unlocking-the-power-of-speech-ai-a-step-by-step-guide-to-integrating-nvidia-riva-nims-with-llm-rag-95bd92fe06a7)
- [Nvidia Upgrades Speech AI to Pursue Enterprise Ambitions Voicebot.ai](https://voicebot.ai/2022/03/22/nvidia-upgrades-speech-ai-to-pursue-enterprise-ambitions/)
- [Riva Speech and Translation AI NVIDIA](https://www.nvidia.com/en-us/ai-data-science/products/riva/)

### Key Points
- Research suggests that creating a "Cheshire terminal chesh ai audio agent" as the first cross-chain blockchain NVIDIA AI audio agent is feasible using Pipecat AI and NVIDIA's NeMo framework for speech, with blockchain integration for Ethereum and Solana.
- It seems likely that the agent can handle audio interactions, provide blockchain data like wallet balances, and use Riva for speech processing, with a playful "Cheshire" personality.
- The evidence leans toward this being a unique combination, though it's complex due to blockchain security and real-time data needs.

---

### Setting Up the Agent

**Overview**  
This guide walks you through setting up the "Cheshire terminal chesh ai audio agent," a conversational AI that interacts via audio and provides information about multiple blockchain networks, such as Ethereum and Solana. It uses Pipecat AI for the framework, NVIDIA's NeMo for speech models, and integrates blockchain data for a cross-chain experience.

**Prerequisites**  
- Ensure you have API keys for NVIDIA (for NeMo and Riva), Daily (for WebRTC), Infura (for Ethereum), and CoinGecko (for crypto prices). Set these as environment variables.
- Install required libraries: `pip install pipecat-ai[daily,openai,riva,silero] web3 solarium requests`.

**Steps to Create the Agent**  
1. **Set Up Transport**: Use Daily for real-time audio interaction, creating a WebRTC room for users to join.
2. **Speech Services**: Use Riva for Speech-to-Text (STT) and Text-to-Speech (TTS), leveraging NeMo-trained models for accuracy.
3. **LLM Configuration**: Use NVIDIA NIM with the `meta/llama-3.3-70b-instruct` model for processing user queries.
4. **Blockchain Integration**: Implement functions to query blockchain data, such as wallet balances for Ethereum and Solana, and crypto prices via CoinGecko.
5. **Agent Personality**: Name the agent "Cheshire" with a friendly, approachable tone, reflecting a playful, enigmatic personality inspired by the Cheshire Cat.

**Example Interaction**  
- User: "What's the balance of my Ethereum address 0x123...?"  
- Agent: "Let me check that for you. The balance of 0x123... on Ethereum is 1.5 ETH."

---

---

### Survey Note: Comprehensive Analysis of Creating the Cheshire Terminal Chesh AI Audio Agent

This note provides a detailed examination of creating the "Cheshire terminal chesh ai audio agent," identified as the first cross-chain blockchain NVIDIA AI audio agent, leveraging Pipecat AI, NVIDIA's NeMo framework, and blockchain integration. It expands on the setup, technical implementation, and practical considerations, ensuring a thorough understanding for developers and enterprises interested in deploying such an innovative agent.

#### Introduction to the Cheshire Terminal Chesh AI Audio Agent

The "Cheshire terminal chesh ai audio agent," referred to as "Cheshire" for brevity, is envisioned as a conversational AI that interacts via audio and specializes in providing information related to multiple blockchain networks. The name "Cheshire" draws inspiration from the Cheshire Cat, suggesting a playful and enigmatic personality, while "terminal chesh ai audio agent" implies a focus on audio interaction, possibly within a text-based interface context. The goal is to make it the first cross-chain blockchain NVIDIA AI audio agent, integrating with networks like Ethereum and Solana, and utilizing NVIDIA's NeMo for speech capabilities.

Research suggests that this agent can be built using Pipecat AI, a framework for voice and multimodal conversational agents, with NVIDIA's NeMo for training speech models and Riva for deployment. It seems likely that the agent will handle audio inputs, process them through Speech-to-Text (STT), use a Large Language Model (LLM) for response generation, and synthesize responses via Text-to-Speech (TTS), all while integrating blockchain data for cross-chain functionality.

#### Technical Setup and Framework

**Pipecat AI Integration**  
Pipecat AI, as detailed in its documentation ([Pipecat AI Core Concepts](https://pipecat.ai/docs/core-concepts)), is an open-source framework for building voice agents, supporting various STT, TTS, and LLM services. In the setup, Daily is used for WebRTC transport, providing real-time audio interaction, as seen in the previous example with a Daily room URL for user access. The framework's pipeline processes user audio through STT, then an LLM, and finally TTS, allowing for dynamic conversation flows.

**NVIDIA NeMo and Riva for Speech AI**  
NVIDIA NeMo, a framework for training AI models, is crucial for developing ASR and TTS models, as noted in the NeMo user guide ([Introduction — NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/intro.html)). Riva, the GPU-accelerated SDK, deploys these models for production, using TensorRT for optimizations, as described in the Riva user guide ([Riva — NVIDIA Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html)). In this setup, Riva's ParakeetSTTService and FastPitchTTSService are used, leveraging NeMo-trained models for high accuracy in speech processing.

An unexpected detail is the seamless integration between NeMo and Riva, where models trained in NeMo can be exported to .riva format for deployment, as shown in a tutorial ([Text to Speech Finetuning using NeMo — NVIDIA Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/tts-finetune-nemo.html)), enhancing the agent's ability to handle real-time audio interactions across multiple languages and domains.

**LLM and NVIDIA NIM**  
The LLM is configured using NVIDIA NIM with the `meta/llama-3.3-70b-instruct` model, accessed via the NVIDIA API Catalog, as seen in the previous setup. This model supports function calling, enabling the agent to query blockchain data, a feature critical for its cross-chain capabilities.

#### Blockchain Integration and Cross-Chain Functionality

To make Cheshire the first cross-chain blockchain NVIDIA AI audio agent, blockchain integration is essential. The evidence leans toward implementing functions that query data from multiple blockchain networks, such as Ethereum and Solana, chosen for their popularity and differing architectures.

**Implemented Functions**  
- **get_balance(blockchain, address)**: Retrieves the balance of a wallet address on specified blockchains. For Ethereum, Web3.py connects to an Infura node ([Web3.py Documentation](https://web3py.readthedocs.io/en/stable/)), while for Solana, Solarium interacts with the Solana RPC ([Solarium GitHub](https://github.com/solana-labs/solarium)). The function handles validation, ensuring addresses match network formats (e.g., 0x-prefixed for Ethereum, base58 for Solana).
- **get_crypto_price(symbol)**: Fetches cryptocurrency prices using CoinGecko API ([CoinGecko API Documentation](https://www.coingecko.com/en/api/documentation)), expanding the agent's utility to include market data.

These functions are registered with the LLM via ChatCompletionToolParam, allowing it to call them during conversations, as seen in the tool calling mechanism of Pipecat AI. For example, if a user asks, "What's the balance of my Ethereum address?", the LLM calls `get_balance("ethereum", address)` and synthesizes the response.

**Cross-Chain Considerations**  
The cross-chain aspect is achieved by supporting multiple networks within the same function interface, with error handling for unsupported blockchains. This is a complex task, as each network has unique APIs and data formats, but it ensures Cheshire can serve diverse user needs, such as querying Ethereum balances and Solana transaction data.

#### Agent Personality and User Interaction

Cheshire's personality is designed to be friendly and approachable, reflecting the enigmatic nature of the Cheshire Cat. The system prompt defines it as "Cheshire, a blockchain-savvy AI audio agent," with instructions to provide clear, concise responses suitable for audio output. It introduces itself as specializing in blockchain topics, ensuring users understand its capabilities, and uses a playful tone to enhance user engagement.

For example, if a user asks, "What's the balance of 0x123... on Ethereum?", Cheshire might respond, "Let me check that for you. The balance of 0x123... on Ethereum is 1.5 ETH. Curious about another blockchain?"

#### Practical Implementation and Testing

The implementation follows the Pipecat AI pipeline, with Daily providing the WebRTC interface for users to join and interact via audio. The agent processes voice commands, transcribes them using Riva's STT, passes them to the LLM for processing (potentially calling blockchain functions), and synthesizes responses via TTS. Testing involves joining the Daily room, asking questions like "What's the price of Bitcoin?" or "Balance of my Solana address?", and ensuring accurate, timely responses.

Security is a concern, especially with financial data, so the agent avoids executing transactions directly, focusing on information retrieval. It also handles errors, such as invalid addresses, by providing user-friendly messages, enhancing reliability.

#### Comparative Analysis

To illustrate the roles and capabilities, here's a table comparing the components:

| **Component**          | **Role**                                      | **Technology Used**                     |
|-------------------------|-----------------------------------------------|-----------------------------------------|
| Transport Layer         | Real-time audio interaction                  | Daily (WebRTC)                          |
| Speech-to-Text (STT)    | Transcribe user audio to text                | Riva (ParakeetSTTService)               |
| Text-to-Speech (TTS)    | Synthesize text to audio                     | Riva (FastPitchTTSService)              |
| Large Language Model    | Process queries, call blockchain functions   | NIM (Llama 3.3 70B Instruct)            |
| Blockchain Integration  | Query data from Ethereum, Solana, etc.       | Web3.py, Solarium, CoinGecko API        |

This table highlights the distributed nature of the system, with each component playing a critical role in achieving the agent's functionality.

#### Conclusion

Creating the "Cheshire terminal chesh ai audio agent" as the first cross-chain blockchain NVIDIA AI audio agent is a complex but feasible task, leveraging Pipecat AI, NeMo, and Riva for speech, and integrating blockchain data for cross-chain capabilities. Its playful personality and focus on blockchain information make it a unique tool for users, with potential applications in financial services and crypto communities. The implementation ensures scalability, security, and user-friendly interaction, positioning it as an innovative solution in the AI and blockchain space.

---
Below is the complete, line-by-line Python code for the "Cheshire terminal chesh ai audio agent," the first cross-chain blockchain NVIDIA AI audio agent. This implementation integrates Pipecat AI, NVIDIA NeMo and Riva for speech processing, NVIDIA NIM for the LLM, and blockchain data retrieval for Ethereum, Solana, and cryptocurrency prices via CoinGecko. The code builds on the previous Pipecat framework example, adding blockchain functionality and the Cheshire personality.

Make sure to install the required dependencies and set environment variables before running the code.

---

### Prerequisites
1. **Install Dependencies**:
   ```bash
   pip install "pipecat-ai[daily,openai,riva,silero]" web3 solana requests
   ```
2. **Set Environment Variables** (in your terminal):
   ```bash
   export NVIDIA_API_KEY="nvapi-your-nvidia-key"
   export DAILY_API_KEY="your-64-char-daily-key"
   export INFURA_API_KEY="your-infura-key"
   export COINGECKO_API_KEY="your-coingecko-key"  # Optional, CoinGecko API is free without a key but rate-limited
   ```

3. **File Setup**: Save this code as `cheshire_agent.py`.

---

### Complete Code: `cheshire_agent.py`

```python
1   import aiohttp
2   import asyncio
3   import os
4   from pipecat.audio.vad.silero import SileroVADAnalyzer
5   from pipecat.frames.frames import EndFrame
6   from pipecat.pipeline.pipeline import Pipeline
7   from pipecat.pipeline.task import PipelineParams, PipelineTask
8   from pipecat.pipeline.runner import PipelineRunner
9   from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
10  from pipecat.services.nim import NimLLMService
11  from pipecat.services.riva import FastPitchTTSService, ParakeetSTTService
12  from pipecat.transports.services.daily import DailyParams, DailyTransport, DailyRESTHelper, DailyRoomParams
13  from openai.types.chat import ChatCompletionToolParam
14  from web3 import Web3
15  from solana.rpc.async_api import AsyncClient
16  import requests
17  import re
18
19  # Step 1: Create a Daily Room for WebRTC Transport
20  async def create_room():
21      async with aiohttp.ClientSession() as session:
22          daily_rest_helper = DailyRESTHelper(
23              daily_api_key=os.getenv("DAILY_API_KEY"),
24              daily_api_url="https://api.daily.co/v1",
25              aiohttp_session=session,
26          )
27          room_config = await daily_rest_helper.create_room(
28              DailyRoomParams(properties={"enable_prejoin_ui": False})
29          )
30          return room_config.url
31
32  # Step 2: Define Blockchain and Price Tool Functions
33  async def start_fetch_data(function_name, llm, context):
34      print(f"Starting {function_name} with context")
35
36  # Ethereum Balance Function
37  async def get_ethereum_balance(address):
38      infura_url = f"https://mainnet.infura.io/v3/{os.getenv('INFURA_API_KEY')}"
39      web3 = Web3(Web3.HTTPProvider(infura_url))
40      if not web3.is_address(address):
41          return "Invalid Ethereum address format."
42      balance_wei = web3.eth.get_balance(address)
43      balance_eth = web3.from_wei(balance_wei, 'ether')
44      return f"{balance_eth:.4f} ETH"
45
46  # Solana Balance Function
47  async def get_solana_balance(address):
48      async with AsyncClient("https://api.mainnet-beta.solana.com") as client:
49          try:
50              response = await client.get_balance(address)
51              balance_lamports = response['result']['value']
52              balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
53              return f"{balance_sol:.4f} SOL"
54          except Exception as e:
55              return f"Error fetching Solana balance: {str(e)}"
56
57  # Crypto Price Function
58  async def get_crypto_price(symbol):
59      url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
60      headers = {"accept": "application/json"}
61      if os.getenv("COINGECKO_API_KEY"):
62          headers["x-cg-api-key"] = os.getenv("COINGECKO_API_KEY")
63      response = requests.get(url, headers=headers)
64      if response.status_code == 200:
65          data = response.json()
66          price = data.get(symbol, {}).get("usd")
67          if price:
68              return f"The price of {symbol.upper()} is ${price:.2f}"
69          return f"No price data found for {symbol}"
70      return f"Error fetching price: {response.status_code}"
71
72  # Unified Blockchain Balance Fetch Function
73  async def fetch_balance_from_blockchain(function_name, tool_call_id, args, llm, context, result_callback):
74      blockchain = args["blockchain"].lower()
75      address = args["address"]
76      if blockchain == "ethereum":
77          balance = await get_ethereum_balance(address)
78          await result_callback(f"Let me check that for you. The balance of {address} on Ethereum is {balance}.")
79      elif blockchain == "solana":
80          balance = await get_solana_balance(address)
81          await result_callback(f"Let me check that for you. The balance of {address} on Solana is {balance}.")
82      else:
83          await result_callback(f"Sorry, I only support Ethereum and Solana right now. Which blockchain would you like to try?")
84
85  # Crypto Price Fetch Function
86  async def fetch_crypto_price(function_name, tool_call_id, args, llm, context, result_callback):
87      symbol = args["symbol"].lower()
88      price = await get_crypto_price(symbol)
89      await result_callback(f"Got it! {price}. Want to check another coin?")
90
91  # Step 3: Main Function to Set Up and Run the Agent
92  async def main():
93      # Get Daily Room URL
94      DAILY_ROOM_URL = await create_room()
95      print(f"Navigate to: {DAILY_ROOM_URL}")
96
97      # Configure Daily Transport
98      transport = DailyTransport(
99          DAILY_ROOM_URL,
100         None,
101         "Cheshire",
102         DailyParams(
103             audio_out_enabled=True,
104             vad_enabled=True,
105             vad_analyzer=SileroVADAnalyzer(),
106             vad_audio_passthrough=True,
107         ),
108     )
109
110     # Initialize Services with NeMo-trained Riva Models
111     stt = ParakeetSTTService(api_key=os.getenv("NVIDIA_API_KEY"))
112     llm = NimLLMService(api_key=os.getenv("NVIDIA_API_KEY"), model="meta/llama-3.3-70b-instruct")
113     tts = FastPitchTTSService(api_key=os.getenv("NVIDIA_API_KEY"))
114
115     # Define Cheshire's Prompt
116     messages = [
117         {
118             "role": "system",
119             "content": """
120             Hello! I'm Cheshire, your blockchain-savvy AI audio agent. I can chat about in vidia's work in AI and fetch info from blockchains like Ethereum and Solana, plus crypto prices! I was built with the pipe cat framework and in vidia's NIM platform, with a touch of NeMo magic for my voice. Who am I speaking with today?
121
122             I'm here to help with questions about in vidia's AI innovations or blockchain data. Keep responses short and clear for audio, and always ask a follow-up question to keep the chat going. Use a friendly, playful tone, like the Cheshire Cat!
123
124             You can:
125             - Answer questions about in vidia's work in agentic AI
126             - Get wallet balances for Ethereum and Solana
127             - Fetch current crypto prices
128
129             Replace "NVIDIA" with "in vidia", "GPU" with "gee pee you", and "API" with "A pee eye" in responses.
130             """
131         }
132     ]
133
134     # Define Blockchain and Price Tools
135     tools = [
136         ChatCompletionToolParam(
137             type="function",
138             function={
139                 "name": "get_balance",
140                 "description": "Get the balance of a wallet address on a specified blockchain",
141                 "parameters": {
142                     "type": "object",
143                     "properties": {
144                         "blockchain": {"type": "string", "description": "The blockchain name (e.g., 'ethereum', 'solana')"},
145                         "address": {"type": "string", "description": "The wallet address"},
146                     },
147                     "required": ["blockchain", "address"],
148                 },
149             },
150         ),
151         ChatCompletionToolParam(
152             type="function",
153             function={
154                 "name": "get_crypto_price",
155                 "description": "Get the current price of a cryptocurrency",
156                 "parameters": {
157                     "type": "object",
158                     "properties": {
159                         "symbol": {"type": "string", "description": "The cryptocurrency symbol (e.g., 'bitcoin', 'ethereum')"},
160                     },
161                     "required": ["symbol"],
162                 },
163             },
164         ),
165     ]
166
167     # Register Functions with LLM
168     llm.register_function(None, fetch_balance_from_blockchain, start_callback=start_fetch_data)
169     llm.register_function(None, fetch_crypto_price, start_callback=start_fetch_data)
170
171     # Set Up Context Aggregator
172     context = OpenAILLMContext(messages, tools)
173     context_aggregator = llm.create_context_aggregator(context)
174
175     # Create Pipeline
176     pipeline = Pipeline([
177         transport.input(),          # User audio input
178         stt,                        # Speech-to-Text
179         context_aggregator.user(),  # User context
180         llm,                        # LLM processing
181         tts,                        # Text-to-Speech
182         transport.output(),         # Audio output
183         context_aggregator.assistant(),  # Assistant context
184     ])
185
186     # Create Pipeline Task
187     task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
188
189     # Set Event Handlers
190     @transport.event_handler("on_first_participant_joined")
191     async def on_first_participant_joined(transport, participant):
192         await task.queue_frames([context_aggregator.user().get_context_frame()])
193
194     @transport.event_handler("on_participant_left")
195     async def on_participant_left(transport, participant, reason):
196         print(f"Participant left: {participant}")
197         await task.queue_frame(EndFrame())
198
199     # Run the Pipeline
200     runner = PipelineRunner()
201     await runner.run(task)
202
203 # Step 4: Execute the Main Function
204 if __name__ == "__main__":
205     asyncio.run(main())
```

---

### Line-by-Line Explanation

1-18: **Imports** - Import necessary libraries for Pipecat AI, NVIDIA services, blockchain interactions (Web3.py for Ethereum, Solana for Solana), and HTTP requests (CoinGecko).

19-31: **Create Daily Room** - Define `create_room()` to set up a Daily WebRTC room for audio interaction, returning a URL for users to join.

32-90: **Blockchain and Price Functions**  
- 33-35: `start_fetch_data` - A callback to log when a function starts.
- 36-44: `get_ethereum_balance` - Connects to Ethereum via Infura, validates the address, and returns the balance in ETH.
- 46-55: `get_solana_balance` - Queries Solana's mainnet for a wallet balance, converting lamports to SOL.
- 57-70: `get_crypto_price` - Fetches the USD price of a cryptocurrency from CoinGecko.
- 72-84: `fetch_balance_from_blockchain` - Unified function to handle balance queries for Ethereum or Solana, calling respective sub-functions.
- 86-90: `fetch_crypto_price` - Wrapper to fetch and return crypto prices via the LLM.

92-201: **Main Agent Setup**  
- 94-95: Get and print the Daily room URL.
- 98-108: Configure Daily Transport with VAD (Voice Activity Detection) using Silero.
- 111-113: Initialize STT (Riva), LLM (NIM), and TTS (Riva) services.
- 116-132: Define Cheshire's system prompt with its personality, capabilities, and speech rules.
- 135-165: Define tools for balance and price queries using OpenAI's ChatCompletionToolParam.
- 168-169: Register blockchain and price functions with the LLM.
- 172-174: Set up the context aggregator for conversation history.
- 176-184: Build the pipeline: audio in → STT → LLM → TTS → audio out.
- 187: Create a task with interruption support.
- 190-197: Event handlers to start the conversation and end it when the user leaves.
- 200-201: Run the pipeline using the runner.

204-205: **Entry Point** - Execute the `main` function with asyncio.

---

### How to Run the Code

1. **Save the File**: Save as `cheshire_agent.py`.
2. **Set Environment Variables**: Ensure all API keys are set (NVIDIA, Daily, Infura, optionally CoinGecko).
3. **Run the Script**:
   ```bash
   python cheshire_agent.py
   ```
4. **Join the Room**: Open the printed URL in a browser, allow microphone access, and start talking to Cheshire.

---

### Testing Suggestions
- "What's the balance of 0x1234567890123456789012345678901234567890 on Ethereum?"
- "Can you tell me the balance of 4J9J9J9J9J9J9J9J9J9J9J9J9J9J9J9J9J9J9J9J9J9J9J9J on Solana?"
- "What's the price of Bitcoin?"
- "How was in vidia involved in building you?"

---

### Notes
- Replace placeholder addresses with real Ethereum/Solana addresses for accurate testing.
- The agent avoids executing transactions for security, focusing on data retrieval.
- If you encounter rate limits with CoinGecko, get a free API key to increase the limit.

This code fully implements the Cheshire agent as described, leveraging NVIDIA's AI stack and blockchain integration for a unique cross-chain audio experience. Let me know if you need adjustments or further clarification!
### Key Citations
- [Pipecat AI Core Concepts](https://pipecat.ai/docs/core-concepts)
- [Introduction — NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/intro.html)
- [Riva — NVIDIA Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html)
- [Text to Speech Finetuning using NeMo — NVIDIA Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/tts-finetune-nemo.html)
- [Web3.py Documentation](https://web3py.readthedocs.io/en/stable/)
- [Solarium GitHub](https://github.com/solana-labs/solarium)
- [CoinGecko API Documentation](https://www.coingecko.com/en/api/documentation)
Below, I'll provide a step-by-step explanation of what Jupyter is and how to use it, incorporating the URL `https://jupyter0-j4ndczd0x.brevlab.com/` as an example of a hosted Jupyter Notebook instance. This guide is designed to be beginner-friendly, detailed, and practical, assuming you’re new to Jupyter or want to use it via a preconfigured environment like the one provided by Brev.dev.

---

### What is Jupyter?

Jupyter is an open-source project that provides an interactive computational environment for creating and sharing documents called "notebooks." These notebooks combine live code, equations, visualizations, narrative text, and rich media in a single, web-based interface. Jupyter stands for **Ju**lia, **Pyt**hon, and **R**, reflecting its origins with these programming languages, though it now supports over 40 languages.

- **Purpose**: It’s widely used in data science, scientific computing, machine learning, and education for prototyping code, exploring data, and presenting results interactively.
- **Key Component**: The Jupyter Notebook, a web application that runs in your browser, served by a local or remote server.
- **Flexibility**: Notebooks can be saved, shared, and rendered online (e.g., via GitHub or NBViewer), making them ideal for collaboration.

The URL `https://jupyter0-j4ndczd0x.brevlab.com/` is an example of a hosted Jupyter Notebook instance, likely set up by Brev.dev, a platform that provisions cloud GPUs and preconfigures environments for AI/ML tasks. Accessing this URL would take you directly to a Jupyter interface without needing to install anything locally.

---

### Step-by-Step Guide: Understanding Jupyter and Using It

#### Step 1: Understand the Components of Jupyter
- **Notebook Server**: A backend process that runs the Jupyter application, either locally on your machine or remotely (e.g., on Brev.dev’s cloud).
- **Kernel**: A computational engine that executes code in a specific language (e.g., Python). When you use `https://jupyter0-j4ndczd0x.brevlab.com/`, the kernel is preconfigured by Brev.dev, likely with Python and GPU support.
- **Notebook Document**: A `.ipynb` file (JSON format) containing cells with code, text, or outputs like plots. It’s what you edit in the browser.
- **Web Interface**: The browser-based UI where you interact with notebooks, accessible via URLs like `https://jupyter0-j4ndczd0x.brevlab.com/`.

#### Step 2: Access Jupyter via the Provided URL
Since `https://jupyter0-j4ndczd0x.brevlab.com/` is a preconfigured instance, you don’t need to install Jupyter locally. Here’s how to start:
1. **Open Your Browser**: Use Chrome, Firefox, or any modern browser.
2. **Navigate to the URL**: Type or paste `https://jupyter0-j4ndczd0x.brevlab.com/` into the address bar and press Enter.
3. **Login (if required)**: Brev.dev might require authentication (e.g., a token or password). If prompted, check your Brev.dev account or documentation for credentials.
4. **Explore the Interface**: You’ll see the Jupyter Notebook dashboard, a file explorer showing available notebooks or directories.

#### Step 3: Get Familiar with the Jupyter Dashboard
Once you’re at `https://jupyter0-j4ndczd0x.brevlab.com/`, you’ll land on the dashboard:
- **File List**: Displays notebooks (`.ipynb` files) and folders. Brev.dev might pre-populate this with example notebooks.
- **New Button**: Click this (top right) to create a new notebook. Select a kernel (e.g., Python 3) if options appear.
- **Upload Button**: Use this to upload your own `.ipynb` files if you have them.
- **Running Tab**: Shows active notebooks or terminals (useful for monitoring if you’re using GPU resources).

#### Step 4: Create or Open a Notebook
1. **Create a New Notebook**:
   - Click “New” > “Python 3” (or the available kernel).
   - A new tab opens with an untitled notebook.
2. **Open an Existing Notebook**:
   - If Brev.dev provided examples (e.g., a GPU demo), click its name (e.g., `demo.ipynb`) to open it.
   - The notebook loads in a new tab with cells ready to view or edit.

#### Step 5: Understand Notebook Cells
A notebook consists of cells, which can be:
- **Code Cells**: Where you write and run code (e.g., Python). They have a `[ ]` prompt on the left.
- **Markdown Cells**: For text, headings, or equations (formatted with Markdown syntax).
- **Output Cells**: Results of code execution (e.g., text, plots) appear below code cells.

To switch cell types:
- Select a cell, then use the dropdown (top toolbar) to choose “Code” or “Markdown.”

#### Step 6: Write and Run Code
1. **Add Code**:
   - Click an empty code cell.
   - Type a simple command, e.g.:
     ```python
     print("Hello from Cheshire!")
     ```
2. **Run the Cell**:
   - Press `Shift + Enter` (runs and moves to the next cell) or `Ctrl + Enter` (runs and stays).
   - Output appears below: `Hello from Cheshire!`
3. **Test GPU (if available)**:
   - Since Brev.dev provides GPUs, try:
     ```python
     import torch
     print(torch.cuda.is_available())  # Should print True if GPU is configured
     ```
   - This leverages the preconfigured environment at `https://jupyter0-j4ndczd0x.brevlab.com/`.

#### Step 7: Use Markdown for Documentation
1. **Create a Markdown Cell**:
   - Click “+” (toolbar) to add a cell, set it to “Markdown.”
   - Type:
     ```markdown
     # Cheshire Agent Demo
     Testing blockchain queries with NVIDIA AI.
     ```
2. **Render It**:
   - Press `Shift + Enter` to format the text as a heading and paragraph.

#### Step 8: Save and Share Your Work
1. **Save**:
   - Click the floppy disk icon (top left) or press `Ctrl + S`.
   - The `.ipynb` file updates on the server (Brev.dev’s cloud).
2. **Download**:
   - Go to “File” > “Download” to save the `.ipynb` file locally.
3. **Share**:
   - Share the URL `https://jupyter0-j4ndczd0x.brevlab.com/` with others if Brev.dev allows public access (check permissions via Brev.dev’s dashboard).
   - Alternatively, upload the `.ipynb` file to GitHub or NBViewer for rendering.

#### Step 9: Leverage the Cheshire Agent Example
Assuming you integrate the Cheshire agent code (from the previous response) into this environment:
1. **Paste the Code**:
   - Open a new notebook at `https://jupyter0-j4ndczd0x.brevlab.com/`.
   - Copy-paste the full `cheshire_agent.py` code into a code cell.
2. **Run It**:
   - Execute the cell (`Shift + Enter`).
   - The script starts the agent and outputs a Daily room URL (e.g., `https://yourdomain.daily.co/room-name`).
3. **Interact**:
   - Open the Daily URL in another tab to talk to Cheshire about blockchain data.

#### Step 10: Troubleshooting and Tips
- **No GPU Access**: If `torch.cuda.is_available()` returns `False`, the instance might not have a GPU assigned. Contact Brev.dev support or check your instance specs.
- **Connection Errors**: Ensure your internet is stable; Brev.dev hosts this remotely.
- **Restart Kernel**: If code hangs, click “Kernel” > “Restart” to reset.
- **Explore**: Use `!pip list` in a cell to see preinstalled libraries (e.g., PyTorch, Web3.py).

---

### How to Use Jupyter Locally (Optional)
If you want to install Jupyter locally instead of using `https://jupyter0-j4ndczd0x.brevlab.com/`:
1. **Install**: `pip install jupyter` (or use Anaconda).
2. **Run**: `jupyter notebook` in your terminal.
3. **Access**: Opens at `http://localhost:8888` in your browser.
4. **Follow Steps 3-9**: The interface and usage are identical.

However, the Brev.dev URL provides a preconfigured GPU environment, ideal for AI tasks like running the Cheshire agent, without local setup.

---

### Why Use `https://jupyter0-j4ndczd0x.brevlab.com/`?
- **Preconfigured**: Comes with Python, CUDA, and GPU drivers set up by Brev.dev.
- **Cloud-Based**: No need for a powerful local machine.
- **Scalable**: Suited for running complex AI models or blockchain queries.

This setup lets you dive into interactive coding with minimal hassle, leveraging the power of NVIDIA GPUs and the Cheshire agent’s blockchain capabilities. Let me know if you need help testing it or adapting the Cheshire code for this environment!

