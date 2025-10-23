# Video Summary Report

**Summary Report – FinSolve AI‑Powered Internal Chatbot Demo**

| Section | What was discussed | Key points & examples |
|---------|--------------------|-----------------------|
| **Purpose of the demo** | Show how an AI chatbot can help employees find information quickly. | • The chatbot answers questions about HR, engineering, finance, marketing, and general company policies. <br>• It uses a large language model (Gemini) to give natural‑language answers. |
| **Role‑based access** | Only people in a certain role can see the data that belongs to that role. | • A “C‑Executive” can see all departments. <br>• A “Finance” user sees only finance data. <br>• A “General” user sees general company policies. |
| **How the chatbot works** | 1. User asks a question. <br>2. The question is turned into a vector (a numeric representation). <br>3. The vector is searched in a vector database that holds all company documents. <br>4. Only the most relevant pieces of text are sent to Gemini. <br>5. Gemini writes a friendly answer that is shown in the chat. | • Uses semantic search – it finds meaning, not just keyword matches. |
| **Tech stack** | • Python (programming language) <br>• Gemini (large language model) <br>• Vector database (for fast semantic search) | • The demo was built in a few hours using open‑source tools. |
| **Content covered by the chatbot** | • **Customer retention** – loyalty programs, targeted email campaigns, gamified features. <br>• **HR policies** – sick leave rules, emergency leave, leave approval process. <br>• **Engineering** – system architecture, microservices, cloud‑native design, security frameworks. <br>• **Finance** – risk mitigation, loss‑reduction strategies, vendor risk management. <br>• **General** – how to apply for leave, legal procedures. | • Example: “What is the procedure to take a leave?” → “Apply via HRMS or leave portal at least three days in advance.” |
| **Demonstrated use‑cases** | 1. **Customer retention** – asked the bot how to strengthen retention; it listed loyalty programs and gamified features. <br>2. **Engineering details** – asked for system architecture; bot described a microservice‑based cloud‑native system. <br>3. **HR policy** – asked about leave procedures; bot gave the exact steps. <br>4. **Finance insights** – asked for ways to overcome loss; bot referenced risk mitigation strategies. | • Each answer was tailored to the user’s role (e.g., a finance user got finance‑specific data). |
| **Benefits highlighted** | • Faster access to accurate information. <br>• Reduces time spent searching documents. <br>• Improves employee productivity. | • “I truly believe this solution will help and boost the productivity for FinSolve technology.” |
| **Next steps / future work** | • Expand the knowledge base to include more documents. <br>• Add more roles (e.g., marketing, legal). <br>• Continuously update policies and procedures. | • Not explicitly mentioned, but implied by the demo’s “future potential.” |

---

### Take‑away Summary (in plain language)

FinSolve built a chatbot that can answer questions about anything inside the company—HR rules, engineering plans, finance risks, or general policies. The bot only shows the information that the person asking the question is allowed to see, so a finance employee can’t accidentally read engineering secrets. When you ask a question, the bot looks up the most relevant parts of the company’s documents, feeds them to an AI model, and gives you a clear, natural‑language answer right away. This saves time, keeps information secure, and helps everyone work more efficiently.