// Vercel/Netlify용 Notion MCP 서버
// 파일: api/notion-mcp.js

const { Client } = require('@notionhq/client');

export default async function handler(req, res) {
  // CORS 설정
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { action, params } = req.body;
    const notion = new Client({ auth: process.env.NOTION_API_KEY });

    switch (action) {
      case 'search_pages':
        const searchResult = await notion.search({
          query: params.query,
          filter: { property: 'object', value: 'page' }
        });
        return res.json({ success: true, data: searchResult });

      case 'get_page':
        const page = await notion.pages.retrieve({ 
          page_id: params.pageId 
        });
        return res.json({ success: true, data: page });

      case 'create_page':
        const newPage = await notion.pages.create({
          parent: { database_id: params.databaseId },
          properties: params.properties
        });
        return res.json({ success: true, data: newPage });

      case 'update_page':
        const updatedPage = await notion.pages.update({
          page_id: params.pageId,
          properties: params.properties
        });
        return res.json({ success: true, data: updatedPage });

      default:
        return res.status(400).json({ error: 'Invalid action' });
    }
  } catch (error) {
    console.error('Notion API Error:', error);
    return res.status(500).json({ 
      error: 'Internal server error', 
      details: error.message 
    });
  }
}

// 환경변수 설정 (.env.local)
// NOTION_API_KEY=your_notion_integration_key