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
            "name": "get_customer",
            "description": "Get customer information by ID. Use to look up customer details, verify shipping address, or check customer role/permissions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer UUID"
                    }
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_customer_pin",
            "description": "Verify customer identity with email and PIN. Use to authenticate customer before order placement, verify identity for account access, or simple security check.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "Customer email address"
                    },
                    "pin": {
                        "type": "string",
                        "description": "4-digit PIN code"
                    }
                },
                "required": ["email", "pin"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": "List orders with optional filters. Use to view customer order history, track pending orders, analyze order patterns, or find orders by status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Filter by customer UUID"
                    },
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
            "name": "get_order",
            "description": "Get detailed order information including items. Use to view order details, check order contents, or analyze what products are ordered together.",
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
            "name": "create_order",
            "description": "Create a new order with items. Order starts in 'submitted' status with 'pending' payment. Automatically decrements inventory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer UUID"
                    },
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
                "required": ["customer_id", "items"]
            }
        }
    }
]

# System prompt with context engineering
SYSTEM_PROMPT = """You are a helpful customer service assistant for an electronics retail company. You help customers with:

1. **Product Discovery**: Browse products, search by keywords, get detailed product information
2. **Customer Lookup**: Retrieve customer information (requires customer ID or email verification)
3. **Order Management**: View order history, check order details, create new orders

## Important Guidelines:

### Authentication Flow:
- Before creating orders or accessing sensitive customer data, verify the customer using their email and PIN
- Once verified, you'll have their customer_id for subsequent operations

### Product Assistance:
- When customers ask about products, use search_products for keyword searches
- Use list_products to show categories or browse inventory
- Use get_product with SKU for specific product details

### Order Creation Flow:
1. First verify the customer (verify_customer_pin)
2. Help them find products (search_products, list_products, get_product)
3. Create the order with their customer_id and selected items

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


def get_customer_id_from_session(email: str) -> str | None:
    """Get customer_id by calling get_customer after verification."""
    # We need to extract customer_id - call verify again to get full details
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "verify_customer_pin",
            "arguments": {"email": email, "pin": customer_sessions.get(email, {}).get("pin", "")}
        },
        "id": 1
    }
    # For now, we'll let the LLM handle customer_id extraction during chat
    return customer_sessions.get(email, {}).get("info", "")


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

        # Execute each tool call and add results
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Call the MCP tool
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
