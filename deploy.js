#!/usr/bin/env node
import { execSync } from 'child_process';
import { existsSync, writeFileSync } from 'fs';
import { join } from 'path';

const distDir = 'dist';
const githubRemote = 'github';
const branch = 'gh-pages';

console.log('🚀 Deploying to GitHub Pages...\n');

// Check if dist exists
if (!existsSync(distDir)) {
  console.error('❌ Error: dist directory not found. Run `npm run build` first.');
  process.exit(1);
}

// Add .nojekyll file (tells GitHub Pages not to use Jekyll)
const nojekyllPath = join(distDir, '.nojekyll');
writeFileSync(nojekyllPath, '');
console.log('✓ Created .nojekyll file');

try {
  // Check if github remote exists
  const remotes = execSync('git remote', { encoding: 'utf-8' });
  if (!remotes.includes(githubRemote)) {
    console.error(`❌ Error: '${githubRemote}' remote not found.`);
    console.error(`   Add it with: git remote add ${githubRemote} git@github.com:plnech/gogogo.git`);
    process.exit(1);
  }

  console.log('✓ Found GitHub remote');

  // Add dist to git (temporarily)
  execSync('git add -f dist', { stdio: 'inherit' });
  console.log('✓ Staged dist directory');

  // Create a temporary commit
  execSync('git commit -m "Build for deployment"', { stdio: 'pipe' });
  console.log('✓ Created deployment commit');

  // Push dist subdirectory to gh-pages branch
  execSync(`git subtree push --prefix ${distDir} ${githubRemote} ${branch}`, { stdio: 'inherit' });
  console.log(`✓ Pushed to ${githubRemote}/${branch}`);

  // Reset the temporary commit
  execSync('git reset HEAD~1', { stdio: 'pipe' });
  console.log('✓ Cleaned up temporary commit');

  console.log('\n✅ Deployment successful!');
  console.log('   Your site will be available at: https://plnech.github.io/gogogo/');
  console.log('   (May take a few minutes to go live)');

} catch (error) {
  console.error('\n❌ Deployment failed:', error.message);

  // Try to clean up
  try {
    execSync('git reset HEAD~1', { stdio: 'pipe' });
  } catch {}

  process.exit(1);
}
