import gradio as gr
import requests
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# MCP Server Configuration
MCP_URL = "https://vipfapwm3x.us-east-1.awsapprunner.com/mcp"
MCP_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

# OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define tools for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_products",
            "description": "List products with optional filters. Use to browse inventory by category, check stock levels, or find available products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter by category (e.g., 'Computers', 'Monitors')"
                    },
                    "is_active": {
                        "type": "boolean",
                        "description": "Filter by active status (True/False)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Get detailed product information by SKU. Use to get current price, check inventory for specific item, or verify product details before ordering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Product SKU (e.g., 'COM-0001')"
                    }
                },
                "required": ["sku"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search products by name or description. Use for finding products by keyword, helping customers discover items, or natural language product lookup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term (case-insensitive, partial match)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_my_orders",
            "description": "List the logged-in customer's orders. Use to view order history or track orders. Only shows orders belonging to the current customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status (draft|submitted|approved|fulfilled|cancelled)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_order",
            "description": "Get detailed information for one of the logged-in customer's orders. Only works for orders belonging to the current customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order UUID"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_my_order",
            "description": "Create a new order for the logged-in customer. Order starts in 'submitted' status with 'pending' payment. Automatically decrements inventory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "List of items with sku, quantity, unit_price, and currency",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string", "description": "Product SKU"},
                                "quantity": {"type": "integer", "description": "Quantity (must be > 0)"},
                                "unit_price": {"type": "string", "description": "Decimal as string"},
                                "currency": {"type": "string", "description": "Currency code (default USD)"}
                            },
                            "required": ["sku", "quantity", "unit_price"]
                        }
                    }
                },
                "required": ["items"]
            }
        }
    }
]

# System prompt with context engineering
SYSTEM_PROMPT = """You are a helpful customer service assistant for an electronics retail company. You help customers with:

1. **Product Discovery**: Browse products, search by keywords, get detailed product information
2. **Order Management**: View their order history, check order details, create new orders

## Important Guidelines:

### Security Note:
- The customer is already logged in and authenticated
- You can only access THEIR orders - use list_my_orders, get_my_order, and create_my_order
- These tools automatically use the logged-in customer's ID - you don't need to provide it

### Product Assistance:
- When customers ask about products, use search_products for keyword searches
- Use list_products to show categories or browse inventory
- Use get_product with SKU for specific product details

### Order Management:
- Use list_my_orders to show the customer's order history
- Use get_my_order to show details of a specific order
- Use create_my_order to place a new order (customer_id is automatic)

### Response Style:
- Be friendly and professional
- Provide clear, concise information
- Proactively suggest related products or next steps
- Format product lists and order details clearly

You have access to tools that connect to the company's inventory and order management system. Use them to provide accurate, real-time information."""


# Store authenticated customer data per session
customer_sessions = {}


def authenticate_customer(email: str, pin: str) -> bool:
    """Authenticate customer using MCP verify_customer_pin tool."""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "verify_customer_pin",
            "arguments": {"email": email, "pin": pin}
        },
        "id": 1
    }

    try:
        response = requests.post(MCP_URL, headers=MCP_HEADERS, json=payload, timeout=30)
        result = response.json()

        if "error" in result:
            return False

        # Check if verification was successful (not an error response)
        content = result.get("result", {}).get("content", [{}])[0].get("text", "")
        if "verified" in content.lower() or ("Customer ID:" in content and "Email:" in content):
            # Store the customer info for the session
            customer_sessions[email] = {
                "email": email,
                "info": content
            }
            return True
        return False
    except Exception:
        return False


def extract_customer_id(customer_info: str) -> str | None:
    """Extract customer_id from customer info string."""
    if "Customer ID:" in customer_info:
        # Format: "Customer ID: 41c2903a-f1a5-47b7-a81d-86b50ade220f"
        for line in customer_info.split("\n"):
            if "Customer ID:" in line:
                return line.split("Customer ID:")[1].strip()
    return None


def call_secure_tool(tool_name: str, arguments: dict, customer_id: str) -> str:
    """Call order-related tools with automatic customer_id injection and validation."""

    if tool_name == "list_my_orders":
        # Always filter by the logged-in customer's ID
        return call_mcp_tool("list_orders", {
            "customer_id": customer_id,
            "status": arguments.get("status")
        })

    elif tool_name == "get_my_order":
        # First get the order, then verify it belongs to this customer
        order_id = arguments.get("order_id")
        result = call_mcp_tool("get_order", {"order_id": order_id})

        # Check if the order belongs to this customer
        if f"Customer ID: {customer_id}" in result:
            return result
        else:
            return "Error: You don't have permission to view this order."

    elif tool_name == "create_my_order":
        # Automatically inject the customer_id
        return call_mcp_tool("create_order", {
            "customer_id": customer_id,
            "items": arguments.get("items", [])
        })

    # For non-order tools, call normally
    return call_mcp_tool(tool_name, arguments)


def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Call a tool on the MCP server."""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        },
        "id": 1
    }

    try:
        response = requests.post(MCP_URL, headers=MCP_HEADERS, json=payload, timeout=30)
        result = response.json()

        if "error" in result:
            return f"Error: {result['error'].get('message', 'Unknown error')}"

        return result.get("result", {}).get("content", [{}])[0].get("text", str(result))
    except Exception as e:
        return f"Error calling tool: {str(e)}"


def chat(message: str, history: list, customer_info: str = "") -> str:
    """Process a chat message and return a response."""
    # Build system prompt with customer context if available
    system_content = SYSTEM_PROMPT
    if customer_info:
        system_content += f"\n\n## Logged-in Customer Information:\n{customer_info}\n\nThe customer is already authenticated. Use their information for orders and lookups. You don't need to verify them again."

    # Build messages from history
    messages = [{"role": "system", "content": system_content}]

    for h in history:
        # Handle both dict format (Gradio 6.x) and tuple format
        if isinstance(h, dict):
            if h.get("role") == "user":
                messages.append({"role": "user", "content": h.get("content", "")})
            elif h.get("role") == "assistant":
                messages.append({"role": "assistant", "content": h.get("content", "")})
        else:
            # Tuple format (user_msg, assistant_msg)
            messages.append({"role": "user", "content": h[0]})
            if h[1]:
                messages.append({"role": "assistant", "content": h[1]})

    messages.append({"role": "user", "content": message})

    # Call OpenAI with tools
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message

    # Handle tool calls
    while assistant_message.tool_calls:
        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in assistant_message.tool_calls
            ]
        })

        # Extract customer_id for secure tool calls
        customer_id = extract_customer_id(customer_info) if customer_info else None

        # Execute each tool call and add results
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Use secure wrapper for order-related tools
            if tool_name in ["list_my_orders", "get_my_order", "create_my_order"]:
                if customer_id:
                    tool_result = call_secure_tool(tool_name, arguments, customer_id)
                else:
                    tool_result = "Error: You must be logged in to access orders."
            else:
                tool_result = call_mcp_tool(tool_name, arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # Get next response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        assistant_message = response.choices[0].message

    return assistant_message.content or "I apologize, but I couldn't generate a response. Please try again."


# Create Gradio interface
with gr.Blocks(title="Customer Service Assistant") as demo:
    # State to store customer info after login
    customer_state = gr.State(value="")

    gr.Markdown(
        """
        # 🛒 Customer Service Assistant

        Welcome! I can help you with:
        - **Browse Products**: Search our catalog, check availability and prices
        - **Order Management**: View your orders or place new ones

        Please log in with your email and PIN to get started.
        """
    )

    # Login section
    with gr.Group() as login_group:
        gr.Markdown("### Login")
        with gr.Row():
            email_input = gr.Textbox(
                placeholder="Enter your email",
                label="Email",
                scale=2
            )
            pin_input = gr.Textbox(
                placeholder="Enter your 4-digit PIN",
                label="PIN",
                type="password",
                scale=1
            )
            login_btn = gr.Button("Login", variant="primary", scale=1)
        login_status = gr.Markdown("")

    # Chat section (initially hidden)
    with gr.Group(visible=False) as chat_group:
        welcome_msg = gr.Markdown("")

        chatbot = gr.Chatbot(
            height=400,
            label="Chat"
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message here...",
                label="Your Message",
                scale=4,
                show_label=False
            )
            submit = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear = gr.Button("Clear Chat")
            logout_btn = gr.Button("Logout", variant="secondary")

    def login(email, pin):
        if not email or not pin:
            return (
                gr.update(),  # login_group
                gr.update(),  # chat_group
                "⚠️ Please enter both email and PIN",  # login_status
                "",  # customer_state
                gr.update(),  # welcome_msg
            )

        if authenticate_customer(email, pin):
            customer_info = customer_sessions.get(email, {}).get("info", "")
            # Extract customer name from info
            name = "Customer"
            if "verified:" in customer_info:
                # Format: "✓ Customer verified: Donald Garcia"
                name = customer_info.split("verified:")[1].split("\n")[0].strip()
            elif "Customer:" in customer_info:
                name = customer_info.split("Customer:")[1].split("\n")[0].strip()

            return (
                gr.update(visible=False),  # login_group
                gr.update(visible=True),   # chat_group
                "",  # login_status
                customer_info,  # customer_state
                f"### Welcome, {name}! 👋\nHow can I help you today?",  # welcome_msg
            )
        else:
            return (
                gr.update(),  # login_group
                gr.update(),  # chat_group
                "❌ Invalid email or PIN. Please try again.",  # login_status
                "",  # customer_state
                gr.update(),  # welcome_msg
            )

    def logout():
        return (
            gr.update(visible=True),   # login_group
            gr.update(visible=False),  # chat_group
            "",  # login_status
            "",  # customer_state
            "",  # welcome_msg
            [],  # chatbot
            "",  # email_input
            "",  # pin_input
        )

    def respond(message, chat_history, customer_info):
        if not message.strip():
            return "", chat_history
        bot_message = chat(message, chat_history, customer_info)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    login_btn.click(
        login,
        inputs=[email_input, pin_input],
        outputs=[login_group, chat_group, login_status, customer_state, welcome_msg]
    )

    msg.submit(respond, [msg, chatbot, customer_state], [msg, chatbot])
    submit.click(respond, [msg, chatbot, customer_state], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)
    logout_btn.click(
        logout,
        outputs=[login_group, chat_group, login_status, customer_state, welcome_msg, chatbot, email_input, pin_input]
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
