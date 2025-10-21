// Google Drive MCP 서버
// 파일: api/drive-mcp.js

const { google } = require('googleapis');

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { action, params } = req.body;
    
    // OAuth2 클라이언트 설정
    const oauth2Client = new google.auth.OAuth2(
      process.env.GOOGLE_CLIENT_ID,
      process.env.GOOGLE_CLIENT_SECRET,
      process.env.GOOGLE_REDIRECT_URI
    );

    oauth2Client.setCredentials({
      access_token: params.access_token || process.env.GOOGLE_ACCESS_TOKEN,
      refresh_token: process.env.GOOGLE_REFRESH_TOKEN
    });

    const drive = google.drive({ version: 'v3', auth: oauth2Client });

    switch (action) {
      case 'search_files':
        const searchResult = await drive.files.list({
          q: params.query,
          fields: 'files(id, name, mimeType, modifiedTime)',
          pageSize: params.limit || 10
        });
        return res.json({ success: true, data: searchResult.data });

      case 'get_file_content':
        const fileContent = await drive.files.get({
          fileId: params.fileId,
          alt: 'media'
        });
        return res.json({ success: true, data: fileContent.data });

      case 'create_file':
        const newFile = await drive.files.create({
          requestBody: {
            name: params.name,
            parents: params.parents
          },
          media: {
            mimeType: params.mimeType,
            body: params.content
          }
        });
        return res.json({ success: true, data: newFile.data });

      default:
        return res.status(400).json({ error: 'Invalid action' });
    }
  } catch (error) {
    console.error('Google Drive API Error:', error);
    return res.status(500).json({ 
      error: 'Internal server error', 
      details: error.message 
    });
  }
}