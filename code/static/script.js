document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');

    // Auto-scroll to bottom
    const scrollToBottom = () => {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    };

    // Format Markdown text robustly
    const formatText = (text) => {
        if (typeof marked !== 'undefined') {
            return marked.parse(text);
        }
        return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
    };

    // Add message to UI
    const addMessage = (text, isUser = false) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${isUser ? 'user' : 'bot'}`;

        msgDiv.innerHTML = `
            <div class="message-avatar">${isUser ? '👤' : '🤖'}</div>
            <div class="message-bubble">${formatText(text)}</div>
        `;

        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    };

    // Create typing indicator
    const createTypingIndicator = () => {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot typing';
        typingDiv.id = 'typing-indicator-msg';
        typingDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-bubble">
                <div class="typing-indicator" style="display: flex;">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatHistory.appendChild(typingDiv);
        scrollToBottom();
    };

    // Remove typing indicator
    const removeTypingIndicator = () => {
        const indicator = document.getElementById('typing-indicator-msg');
        if (indicator) {
            indicator.remove();
        }
    };

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const message = userInput.value.trim();
        if (!message) return;

        // Disable input
        userInput.value = '';
        userInput.disabled = true;

        // Add user message
        addMessage(message, true);

        // Show typing
        createTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();

            removeTypingIndicator();

            if (response.ok) {
                addMessage(data.response);
            } else {
                addMessage(`Error: ${data.error}`);
            }
        } catch (error) {
            removeTypingIndicator();
            addMessage('Network error. Could not reach the server.');
        } finally {
            // Re-enable input
            userInput.disabled = false;
            userInput.focus();
        }
    });
});
