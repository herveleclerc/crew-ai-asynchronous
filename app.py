from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from langchain.tools import tool
from crewai.tasks.task_output import TaskOutput
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from textwrap import dedent
from googletrans import Translator, constants
import os


load_dotenv(dotenv_path='./.env')


def get_llm():
  llm = AzureChatOpenAI(
        model           = os.getenv("AZURE_MODEL"),
        deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME"),
        api_key         = os.getenv("AZURE_API_KEY"),
        azure_endpoint  = os.getenv("AZURE_ENDPOINT"),
        api_version     = os.getenv("AZURE_API_VERSION"),
        )
  return llm


llm=get_llm()

search_tool = DuckDuckGoSearchRun()

# Define the topic of interest
#topic = 'AI driven autoscaling in kubernetes'

topic = 'The future of AI'

# Output lang
lang = 'french'

# Loading Human Tools
human_tools = load_tools(["human"])

def callback_function(output: TaskOutput):
    # Do something after the task is completed
    # Example: Send an email to the manager
    print(f"""
        Task completed!
        Task: {output.description}
        Output: {output.result}
    """)
    
# Creating custom tools
class ContentTools:
    @tool("Read webpage content")
    def read_content(url: str) -> str:
        """Read content from a webpage."""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = soup.get_text()
        return text_content[:5000]
    
class TranslateTools():
    @tool("Translate Content")
    def translate_content(content: str) -> str:
        """Translate content in french"""
        translation = translator.translate(content, dest="fr")
        return (translation)

# Define the manager agent
manager = Agent(
    role='Project Manager',
    goal='Coordinate the project to ensure a seamless integration of research findings into compelling narratives',
    verbose=False,
    backstory=dedent("""With a strategic mindset and a knack for leadership, you excel at guiding teams towards
    their goals, ensuring projects not only meet but exceed expectations."""),
    allow_delegation=True,
    max_iter=10,
    max_rpm=20,
    llm=llm
)

# Define the senior researcher agent
researcher = Agent(
    role='Senior Researcher',
    goal=f'Uncover groundbreaking technologies around {topic}',
    verbose=False,
    backstory=dedent("""Driven by curiosity, you're at the forefront of innovation, eager to explore and share
    knowledge that could change the world."""),
    llm=llm
)

# Define the writer agent
writer = Agent(
    role='Writer',
    goal=f'Narrate compelling tech stories around {topic}',
    verbose=False,
    backstory=dedent("""With a flair for simplifying complex topics, you craft engaging narratives that captivate
    and educate, bringing new discoveries to light in an accessible manner."""),
    llm=llm
)

# Define the translator agent
translator = Agent(
    role='Translator',
    goal=f'You are a helpful assistant that translates english text to {lang}',
    verbose=False,
    backstory=f'You are a helpful assistant that translates english to {lang}.',
    llm=llm
)

# Define the asynchronous research tasks
list_ideas = Task(
    description=f"List of 10 interesting ideas to explore for an article about {topic}.",
    expected_output="Bullet point list of 10 ideas for an article.",
    tools=[search_tool, ContentTools().read_content],  
    agent=researcher,
    async_execution=True
)

list_important_history = Task(
    description=f"Research the history of {topic} and identify the 10 most important events.",
    expected_output="Bullet point list of 10 important events.",
    tools=[search_tool, ContentTools().read_content],
    agent=researcher,
    async_execution=True
)

list_tendency = Task(
    description=f"Searches for trend and progress {topic} and identifies the 5  most disruptive trends",
    expected_output="Bullet point list of 5 more revelant isruptive trends.",
    tools=[search_tool, ContentTools().read_content],
    agent=researcher,
    async_execution=True
)

# Define the writing task that waits for the outputs of the two research tasks
write_article = Task(
    description=f"Compose an insightful article on {topic}, including its history and the latest interesting ideas and actual solutions.",
    expected_output=f"A 1000 words article on {topic} structured in 4 paragraphs, each with a summary title.",
    tools=[search_tool, ContentTools().read_content],  
    agent=writer,
    context=[list_ideas, list_important_history],  # Depends on the completion of the two asynchronous tasks
    callback=callback_function
)

# Define the manager's coordination task
manager_task = Task(
    description=dedent(f"""Oversee the integration of research findings and narrative development to produce a final comprehensive
    report on {topic}. Ensure the research is accurately represented and the narrative is engaging and informative."""),
    expected_output=f'A final comprehensive article that combines the research findings and narrative and tendency on {topic}.',
    agent=manager
)

translator_task = Task(
    description=f"""Translate the text in a technical {lang}.""",
    expected_output=f'A perfect text written in {lang}',
    agent=manager,
    tools=[TranslateTools().translate_content],
    context=[write_article],  
)

# Forming the crew with a hierarchical process including the manager
crew = Crew(
    agents=[manager, researcher, writer, translator],
    tasks=[list_ideas, list_important_history, list_tendency, write_article, manager_task,translator_task],
    process=Process.hierarchical,
    manager_llm=llm
)

# Kick off the crew's work
results = crew.kickoff()

# Print the results
print("Crew Work Results:")
print(results)
