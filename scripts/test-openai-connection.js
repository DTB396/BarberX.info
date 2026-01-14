#!/usr/bin/env node
// Quick test script to verify OpenAI API connection
// Usage: node scripts/test-openai-connection.js

import OpenAI from 'openai';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Check for API key
if (!process.env.OPENAI_API_KEY) {
  console.error('❌ Error: OPENAI_API_KEY environment variable is not set.');
  console.error('Please set it in a .env file or as an environment variable.');
  process.exit(1);
}

console.log('✅ OpenAI API key is configured');
console.log(`   Key starts with: ${process.env.OPENAI_API_KEY.substring(0, 8)}...`);
console.log('');

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Test the connection with a simple request
async function testConnection() {
  try {
    console.log('🔍 Testing OpenAI API connection...');
    console.log('   Sending a minimal test request...');
    
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'user',
          content: 'Say "API connection successful!" in exactly 5 words.'
        }
      ],
      max_tokens: 20,
      temperature: 0,
    });
    
    const result = response.choices[0].message.content;
    console.log('');
    console.log('✅ SUCCESS! OpenAI API is working correctly.');
    console.log(`   Response: ${result}`);
    console.log('');
    console.log('📊 Usage:');
    console.log(`   Prompt tokens: ${response.usage.prompt_tokens}`);
    console.log(`   Completion tokens: ${response.usage.completion_tokens}`);
    console.log(`   Total tokens: ${response.usage.total_tokens}`);
    console.log('');
    console.log('ℹ️  Your OpenAI account information:');
    console.log('   Model: gpt-4o-mini');
    console.log('   Rate limits (free tier): 3 requests/minute, 100,000 tokens/minute');
    console.log('   Check usage at: https://platform.openai.com/usage');
    console.log('');
    console.log('🎉 You\'re ready to analyze cases!');
    console.log('   Run: node scripts/analyze-cases.js');
    
  } catch (error) {
    console.error('');
    console.error('❌ ERROR: Failed to connect to OpenAI API');
    console.error('');
    
    if (error.status === 401) {
      console.error('Authentication error - Invalid API key');
      console.error('  • Check that your API key is correct');
      console.error('  • Get a new key at: https://platform.openai.com/api-keys');
    } else if (error.status === 429) {
      console.error('Rate limit exceeded');
      console.error('  • Free tier: 3 requests/minute, 100,000 tokens/minute');
      console.error('  • Wait a minute and try again');
      console.error('  • Consider upgrading: https://platform.openai.com/account/billing');
    } else if (error.status === 402) {
      console.error('Payment required - Insufficient credits');
      console.error('  • Add credits at: https://platform.openai.com/account/billing');
    } else if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
      console.error('Network error - Cannot reach OpenAI servers');
      console.error('  • Check your internet connection');
      console.error('  • Check OpenAI status: https://status.openai.com/');
    } else {
      console.error(`HTTP ${error.status || 'N/A'}: ${error.message}`);
    }
    
    console.error('');
    console.error('Full error details:');
    console.error(error);
    process.exit(1);
  }
}

// Run the test
testConnection();
